#pragma once

// The wave plan of a ubatch that runs in waves. A pure function of the
// call's expert ids, which experts are already in RAM and the graph's fixed shape (W waves, S slots per wave, cap pairs per wave);
// no engine state, so it is tested on its own.
//
// Built once per (layer, call) at wave 0 and stored; every later wave reads it. Each expert sits in one wave, or - when its pairs
// overflow the wave's cap - in two consecutive waves (it stays in its slot, pinned, for both). Every pair (token, k) of an expert the
// plan holds is in exactly one wave.
// The engine passes no cap (n_tok x k: each expert in one wave, order A always fits); a cap, the split and order B
// are reached only through the test hook MOE_STREAM_TEST_WAVE_SLACK and the standalone test.
//
// Two orders, tried in this order:
//  A (residents first): the experts already in RAM, then the misses, each group in first-use order; waves filled one after the other
//    up to S experts or cap pairs, the boundary expert's pairs continuing in the next wave. A needed resident is never evicted before
//    its wave: every wave that reads (has a miss) comes after every wave that holds a resident, except the one wave that has both,
//    whose residents are pinned before its reads. In a prompt's last ubatch (prompt-end weights given) residents still
//    go first, each group in ascending weight, so the last wave holds the most recent misses; and the P residents with the highest
//    weight stay pinned to the end of the call (the engine re-pins them in every later wave), P = floor((W x S - |U|) / (W - 1)):
//    wave 0 holds S experts, every later wave S - P misses + the P pins, so the plan still fits in W waves. They survive the call's
// evictions, so the reply starts with them in RAM.
//  B (balanced, when A needs more than W waves): the experts by pair count, largest first, dealt to the waves in snake order
//    (0..W-1, W-1..0, ...). Per wave at most ceil(|U| / W) <= S experts, and at most ceil(P / W) + the largest pair count pairs
//    (the waves' pair sums never differ by more than the largest item in a snake deal of sorted items), so with
//    cap >= ceil(8 n_tok / W) + n_tok it always fits, with no split.

#include <algorithm>
#include <climits>
#include <cstdint>
#include <string>
#include <vector>

struct moe_wave_plan {
    int32_t n_waves   = 0;       // W, the graph's wave count
    int64_t cap       = 0;       // pairs per wave the index tensors hold
    int64_t n         = 0;       // pairs of the call (n_tok x k)
    bool    res_first = false;   // order A fitted
    int32_t n_split   = 0;       // experts in two waves
    int32_t n_pin     = 0;       // residents pinned from their wave to the call's end (prompt-end calls only)
    int32_t pin_max   = 0;       // the bound P = floor((W x S - |U|) / (W - 1)) the plan used (0 = no pins)

    std::vector<int32_t> cnt;      // [n_expert] pairs of e in the call (0 = not in the plan)
    std::vector<int32_t> w_first;  // [n_expert] the first wave of e, -1 = not in the plan
    std::vector<int32_t> n_first;  // [n_expert] pairs of e in its first wave (the rest are in w_first + 1)
    std::vector<int32_t> wave_off; // [W + 1] offsets into wave_e
    std::vector<int32_t> wave_e;   // the experts of each wave (a split expert in both)
    std::vector<int64_t> wave_p;   // [W] pairs per wave
    std::vector<int16_t> pair_wave;// [n] the wave of pair i, -1 = its expert is not in the plan (the VRAM pass serves it)
    std::vector<uint8_t> pin;      // [n_expert] 1 = pinned after its own wave(s) to the end of the call

    bool in_wave(int32_t e, int32_t w) const {
        const int32_t w0 = w_first[e];
        return w0 >= 0 && (w0 == w || (w0 + 1 == w && n_first[e] < cnt[e]));
    }
    // the last wave whose pairs e has (w_first, or w_first + 1 for a split expert)
    int32_t w_last(int32_t e) const { return w_first[e] + (n_first[e] < cnt[e] ? 1 : 0); }
    // E holds its slot in wave w only as a pin (after its own wave)
    bool pinned_in(int32_t e, int32_t w) const { return !pin.empty() && pin[e] && w_first[e] >= 0 && w > w_last(e); }
};

// the cap of a wave's index tensors: room for the average plus one whole expert, so order B always fits; never below n_tok (one
// expert can have a pair in every token); slack < 0 = the safe default (n_tok)
static inline int64_t moe_wave_cap(int64_t n_tok, int64_t k, int32_t n_waves, int64_t slack) {
    const int64_t w   = std::max<int32_t>(1, n_waves);
    const int64_t avg = (n_tok*k + w - 1)/w;
    return std::max<int64_t>(n_tok, avg + (slack < 0 ? n_tok : slack));
}

// uniq: the call's experts that need a RAM slot, in first-use order; ids: [n_tok x k]; res[e] != 0: e is in RAM (or being read);
// pe_w: per expert prompt-end weight, nullptr = none. Returns false only if neither order fits (possible only with a cap below the
// safe one); err says why.
static inline bool moe_wave_plan_build(moe_wave_plan & p, const std::vector<int32_t> & uniq, const int32_t * ids, int64_t n,
        const uint8_t * res, const float * pe_w, int32_t n_waves, int64_t width, int64_t cap, int32_t n_expert,
        std::string & err) {
    p.n_waves   = n_waves;
    p.cap       = cap;
    p.n         = n;
    p.res_first = false;
    p.n_split   = 0;
    p.n_pin     = 0;
    p.pin_max   = 0;
    p.pin    .assign((size_t) n_expert, 0);
    p.cnt    .assign((size_t) n_expert, 0);
    p.w_first.assign((size_t) n_expert, -1);
    p.n_first.assign((size_t) n_expert, 0);
    p.wave_p .assign((size_t) n_waves, 0);
    p.pair_wave.assign((size_t) n, -1);

    for (const int32_t e : uniq) {
        p.cnt[e] = -1;   // a mark: only the plan's experts are counted
    }
    for (int64_t i = 0; i < n; i++) {
        const int32_t e = ids[i];
        if (e >= 0 && e < n_expert && p.cnt[e] != 0) {
            p.cnt[e] += p.cnt[e] < 0 ? 2 : 1;   // -1 -> 1 on the first pair
        }
    }

    if (n_waves > INT16_MAX) {
        err = std::to_string(n_waves) + " waves do not fit the pair table's 16-bit wave numbers";
        return false;
    }

    // order A: the residents, then the misses; with prompt-end weights each group in ascending weight
    std::vector<int32_t> ord;
    ord.reserve(uniq.size());
    for (int pass = 0; pass < 2; pass++) {
        for (const int32_t e : uniq) {
            if ((res[e] != 0) == (pass == 0)) {
                ord.push_back(e);
            }
        }
    }
    const size_t n_res = (size_t) (std::find_if(ord.begin(), ord.end(), [&](int32_t e) { return res[e] == 0; }) - ord.begin());
    if (pe_w) {
        const auto by_w = [&](int32_t a, int32_t c) { return pe_w[a] < pe_w[c]; };
        std::stable_sort(ord.begin(), ord.begin() + (std::ptrdiff_t) n_res, by_w);
        std::stable_sort(ord.begin() + (std::ptrdiff_t) n_res, ord.end(), by_w);
        // The residents with the highest weight (the last ones of the ascending group) stay pinned to the call's end, at most
        // P = floor((W x S - |U|) / (W - 1)) so that wave 0's S + (W - 1) later waves of S - P still hold every expert
        if (n_waves > 1) {
            const int64_t room = (int64_t) n_waves*width - (int64_t) uniq.size();
            p.pin_max = (int32_t) std::max<int64_t>(0, room/(n_waves - 1));
            p.n_pin   = (int32_t) std::min<int64_t>(p.pin_max, (int64_t) n_res);
            for (size_t i = n_res - (size_t) p.n_pin; i < n_res; i++) {
                p.pin[ord[i]] = 1;
            }
        }
    }
    {
        int32_t w  = 0;
        int64_t nw = 0;   // experts in wave w
        int64_t pw = 0;   // pairs in wave w
        bool    ok = true;
        for (size_t i = 0; i < ord.size() && ok; i++) {
            const int32_t e   = ord[i];
            int64_t       rem = p.cnt[e];
            bool          first = true;
            while (rem > 0) {
                if (nw == (w == 0 ? width : width - p.n_pin) || pw == cap) {   // a later wave's places minus the pins
                    w++;
                    nw = 0;
                    pw = 0;
                }
                if (w >= n_waves) {
                    ok = false;
                    break;
                }
                const int64_t take = std::min(rem, cap - pw);
                if (first) {
                    p.w_first[e] = w;
                    p.n_first[e] = (int32_t) take;
                    first        = false;
                } else {
                    p.n_split += 1;   // the second part, in wave w_first + 1 (take <= cnt <= n_tok <= cap: no third part)
                }
                p.wave_p[w] += take;
                nw  += 1;
                pw  += take;
                rem -= take;
            }
        }
        p.res_first = ok;
    }

    if (!p.res_first) {   // order B
        std::fill(p.w_first.begin(), p.w_first.end(), -1);
        std::fill(p.n_first.begin(), p.n_first.end(), 0);
        std::fill(p.wave_p .begin(), p.wave_p .end(), 0);
        std::fill(p.pin    .begin(), p.pin    .end(), 0);   // no pins in the balanced order
        p.n_split = 0;
        p.n_pin   = 0;
        p.pin_max = 0;
        ord = uniq;
        std::stable_sort(ord.begin(), ord.end(), [&](int32_t a, int32_t c) { return p.cnt[a] > p.cnt[c]; });
        std::vector<int64_t> nw((size_t) n_waves, 0);
        for (size_t i = 0; i < ord.size(); i++) {
            const int64_t r  = (int64_t) (i/(size_t) n_waves);
            const int64_t c  = (int64_t) (i%(size_t) n_waves);
            const int32_t w  = (int32_t) (r % 2 == 0 ? c : n_waves - 1 - c);
            const int32_t e  = ord[i];
            p.w_first[e] = w;
            p.n_first[e] = p.cnt[e];
            p.wave_p[w] += p.cnt[e];
            nw[w]       += 1;
        }
        for (int32_t w = 0; w < n_waves; w++) {
            if (nw[w] > width || p.wave_p[w] > cap) {
                err = "wave " + std::to_string(w) + " holds " + std::to_string(nw[w]) + " experts (width " + std::to_string(width) +
                      ") and " + std::to_string(p.wave_p[w]) + " pairs (cap " + std::to_string(cap) + ") in both orders";
                return false;
            }
        }
    }

    // the wave lists and every pair's wave (a split expert: its first n_first pairs in token order in the first wave)
    p.wave_off.assign((size_t) n_waves + 1, 0);
    for (const int32_t e : uniq) {
        p.wave_off[(size_t) p.w_first[e] + 1] += 1;
        if (p.n_first[e] < p.cnt[e]) {
            p.wave_off[(size_t) p.w_first[e] + 2] += 1;
        }
    }
    for (int32_t w = 0; w < n_waves; w++) {
        p.wave_off[(size_t) w + 1] += p.wave_off[(size_t) w];
    }
    p.wave_e.assign((size_t) p.wave_off[(size_t) n_waves], -1);
    std::vector<int32_t> fill(p.wave_off.begin(), p.wave_off.end() - 1);
    for (const int32_t e : ord) {   // in plan order, so a wave lists its experts as they were packed
        p.wave_e[(size_t) fill[(size_t) p.w_first[e]]++] = e;
        if (p.n_first[e] < p.cnt[e]) {
            p.wave_e[(size_t) fill[(size_t) p.w_first[e] + 1]++] = e;
        }
    }
    std::vector<int32_t> seen((size_t) n_expert, 0);
    for (int64_t i = 0; i < n; i++) {
        const int32_t e = ids[i];
        if (e >= 0 && e < n_expert && p.w_first[e] >= 0) {
            p.pair_wave[(size_t) i] = (int16_t) (p.w_first[e] + (seen[e]++ < p.n_first[e] ? 0 : 1));
        }
    }
    return true;
}

// the plan's own rules, checked from scratch (the engine runs it under --moe-stream-slot-check; the standalone test runs it on every
// case): every pair of a planned expert in exactly one wave in [0, W), every other pair in none; per wave <= width experts (the
// wave's own + the pins held after their wave) and <= cap pairs; an expert in at most two consecutive waves; the lists match the pairs;
// The pins are planned experts, n_pin of them, at most the bound
static inline bool moe_wave_plan_check(const moe_wave_plan & p, const std::vector<int32_t> & uniq, const int32_t * ids, int64_t n,
        int64_t width, int32_t n_expert, std::string & err) {
    std::vector<uint8_t> want((size_t) n_expert, 0);
    for (const int32_t e : uniq) {
        want[e] = 1;
    }
    std::vector<int64_t> pw((size_t) p.n_waves, 0);
    std::vector<int32_t> lo((size_t) n_expert, INT32_MAX), hi((size_t) n_expert, -1);
    for (int64_t i = 0; i < n; i++) {
        const int32_t e = ids[i];
        const int32_t w = p.pair_wave[(size_t) i];
        if (e < 0 || e >= n_expert || !want[e]) {
            if (w != -1) {
                err = "pair " + std::to_string(i) + " of an expert outside the plan has wave " + std::to_string(w);
                return false;
            }
            continue;
        }
        if (w < 0 || w >= p.n_waves) {
            err = "pair " + std::to_string(i) + " (expert " + std::to_string(e) + ") has no wave";
            return false;
        }
        if (!p.in_wave(e, w)) {
            err = "pair " + std::to_string(i) + " is in wave " + std::to_string(w) + " but expert " + std::to_string(e) + " is not";
            return false;
        }
        pw[(size_t) w] += 1;
        lo[e] = std::min(lo[e], w);
        hi[e] = std::max(hi[e], w);
    }
    // The pins
    int32_t n_pin = 0;
    std::vector<int64_t> n_pinned((size_t) p.n_waves, 0);   // per wave: pins held after their own wave
    for (int32_t e = 0; e < n_expert && !p.pin.empty(); e++) {
        if (!p.pin[e]) {
            continue;
        }
        if (!want[e] || p.w_first[e] < 0) {
            err = "pinned expert " + std::to_string(e) + " is not in the plan";
            return false;
        }
        n_pin += 1;
        for (int32_t w = p.w_last(e) + 1; w < p.n_waves; w++) {
            n_pinned[(size_t) w] += 1;
        }
    }
    if (n_pin != p.n_pin || p.n_pin > p.pin_max) {
        err = std::to_string(n_pin) + " pinned experts, the plan says " + std::to_string(p.n_pin) + " (bound " + std::to_string(p.pin_max) + ")";
        return false;
    }
    for (int32_t w = 0; w < p.n_waves; w++) {
        const int64_t ne = p.wave_off[(size_t) w + 1] - p.wave_off[(size_t) w] + n_pinned[(size_t) w];
        if (ne > width || pw[(size_t) w] > p.cap || pw[(size_t) w] != p.wave_p[(size_t) w]) {
            err = "wave " + std::to_string(w) + ": " + std::to_string(ne) + " experts (width " + std::to_string(width) + "), " +
                  std::to_string(pw[(size_t) w]) + " pairs (cap " + std::to_string(p.cap) + ", plan says " +
                  std::to_string(p.wave_p[(size_t) w]) + ")";
            return false;
        }
        for (int32_t i = p.wave_off[(size_t) w]; i < p.wave_off[(size_t) w + 1]; i++) {
            if (!p.in_wave(p.wave_e[(size_t) i], w)) {
                err = "wave " + std::to_string(w) + " lists expert " + std::to_string(p.wave_e[(size_t) i]) + " that is not in it";
                return false;
            }
        }
    }
    // both directions: every planned expert of a wave is in its list (the width check above counts the list)
    std::vector<int64_t> n_in((size_t) p.n_waves, 0);
    for (const int32_t e : uniq) {
        for (int32_t w = std::max(0, p.w_first[e]); w <= std::min(p.n_waves - 1, p.w_first[e] + 1); w++) {
            n_in[(size_t) w] += p.in_wave(e, w) ? 1 : 0;
        }
    }
    for (int32_t w = 0; w < p.n_waves; w++) {
        if (n_in[(size_t) w] != p.wave_off[(size_t) w + 1] - p.wave_off[(size_t) w]) {
            err = "wave " + std::to_string(w) + " has " + std::to_string(n_in[(size_t) w]) + " experts but lists " +
                  std::to_string(p.wave_off[(size_t) w + 1] - p.wave_off[(size_t) w]);
            return false;
        }
    }
    for (const int32_t e : uniq) {
        if (hi[e] < 0 || hi[e] - lo[e] > 1) {
            err = "expert " + std::to_string(e) + " spans waves " + std::to_string(lo[e]) + ".." + std::to_string(hi[e]);
            return false;
        }
    }
    return true;
}

// the three index tensors of wave w (4.1): one row per pair of the wave, in token order - slot_w = the expert's slot (slot_of[e]),
// tok_w = the token row, dst_w = the pair's row t x k + j of the layer's [n_embd, k x n_tok] output; then padding up to cap rows:
// slot_w = zero_slot (an all-zero slice), tok_w = 0, dst_w = n (the one spare row). Returns the rows used (<= cap: the plan's rule).
static inline int64_t moe_wave_rows(const moe_wave_plan & p, int32_t w, const int32_t * ids, const int32_t * slot_of,
        int32_t zero_slot, int64_t k, int32_t * slot_w, int32_t * tok_w, int32_t * dst_w) {
    int64_t r = 0;
    for (int64_t i = 0; i < p.n && r < p.cap; i++) {
        if (p.pair_wave[(size_t) i] == w) {
            slot_w[r] = slot_of[ids[i]];
            tok_w [r] = (int32_t) (i/k);
            dst_w [r] = (int32_t) i;
            r++;
        }
    }
    for (int64_t i = r; i < p.cap; i++) {
        slot_w[i] = zero_slot;
        tok_w [i] = 0;
        dst_w [i] = (int32_t) p.n;
    }
    return r;
}
