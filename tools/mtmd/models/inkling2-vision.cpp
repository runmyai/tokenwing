#include "models.h"

// Inkling-2-Mini BF16 Vision Encoder (`model.visual.*`):
//   - 23-layer ViT (`hidden_size = 1024`, `n_head = 16`, `d_head = 64`, `intermediate_size = 4096`, `eps = 1e-5`)
//   - 40x40 tiles split into 2x2 patches of 20x20x3 (`1200`-dim channel-last RGB linear patch embedding)
//   - Bilinearly interpolated `32x32` learned position embeddings + prepended `CLS + cls_pos_embd` token at `(0, 0)`
//   - `pre_ln` -> 23 blocks with 2D axial `GGML_ROPE_TYPE_NORM` (`wpos + 1` in `[0..31]`, `hpos + 1` in `[32..63]`),
//     non-causal full attention, `ls1`/`ls2` layer scales, and exact ERF-GELU MLP
//   - Strip CLS token 0 -> 2x2 spatial merge (`4096`) -> `mm.0 -> FFN_GELU_ERF -> mm.2` (`2304` text hidden dim)
ggml_cgraph * clip_graph_inkling2::build() {
    GGML_ASSERT(model.patch_embeddings_0 != nullptr);
    GGML_ASSERT(model.position_embeddings != nullptr);
    GGML_ASSERT(model.class_embedding != nullptr);
    GGML_ASSERT(model.cls_pos_embedding != nullptr);
    GGML_ASSERT(model.pre_ln_w != nullptr && model.pre_ln_b != nullptr);
    GGML_ASSERT(model.mm_0_w != nullptr && model.mm_2_w != nullptr);

    const int merge = hparams.n_merge; // 2
    GGML_ASSERT(merge == 2);
    GGML_ASSERT(n_patches_x % merge == 0 && n_patches_y % merge == 0);

    const int rows  = n_patches_y / merge;
    const int cols  = n_patches_x / merge;
    const int n_tok = n_patches;       // rows * cols * 4
    const int n_seq = 1 + n_tok;       // CLS + n_tok
    const int n_out = rows * cols;     // n_tok / 4

    auto inp_i32 = [&](const char * name, int64_t n) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n);
        ggml_set_name(t, name);
        ggml_set_input(t);
        return t;
    };

    auto inp_f32_2d = [&](const char * name, int64_t ne0, int64_t ne1) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, ne0, ne1);
        ggml_set_name(t, name);
        ggml_set_input(t);
        return t;
    };

    // 1. Reorder raw CHW pixels `[nx, ny, 3, 1]` into `(c=3, px=20, py=20)` x `(mw=2, mh=2, tx=cols, ty=rows)`
    //    matching `Inkling2ImagePreproc` + `_Ink2VisionEmbeddings` (`[1200, n_tok]`)
    ggml_tensor * inp_raw = build_inp_raw(3);
    ggml_tensor * p = ggml_permute(ctx0, inp_raw, 1, 2, 0, 3);
    p = ggml_cont_4d(ctx0, p, 3 * patch_size, merge * cols, patch_size, merge * rows);
    p = ggml_permute(ctx0, p, 0, 2, 1, 3);
    p = ggml_cont_4d(ctx0, p, 3 * patch_size * patch_size * merge, cols, merge, rows);
    p = ggml_permute(ctx0, p, 0, 2, 1, 3);
    p = ggml_cont_2d(ctx0, p, 3 * patch_size * patch_size, n_tok);

    ggml_tensor * patch_emb = build_mm(model.patch_embeddings_0, p); // [1024, n_tok]

    // 2. Exact 4-tap bilinear position embedding gather + weighted sum (`_vision_interp_indices_and_weights`)
    ggml_tensor * pos_idx0 = inp_i32("ink2_pos_idx0", n_tok);
    ggml_tensor * pos_idx1 = inp_i32("ink2_pos_idx1", n_tok);
    ggml_tensor * pos_idx2 = inp_i32("ink2_pos_idx2", n_tok);
    ggml_tensor * pos_idx3 = inp_i32("ink2_pos_idx3", n_tok);

    ggml_tensor * pos_w0 = inp_f32_2d("ink2_pos_w0", 1, n_tok);
    ggml_tensor * pos_w1 = inp_f32_2d("ink2_pos_w1", 1, n_tok);
    ggml_tensor * pos_w2 = inp_f32_2d("ink2_pos_w2", 1, n_tok);
    ggml_tensor * pos_w3 = inp_f32_2d("ink2_pos_w3", 1, n_tok);

    ggml_tensor * pos_emb = ggml_mul(ctx0, ggml_get_rows(ctx0, model.position_embeddings, pos_idx0), pos_w0);
    pos_emb = ggml_add(ctx0, pos_emb, ggml_mul(ctx0, ggml_get_rows(ctx0, model.position_embeddings, pos_idx1), pos_w1));
    pos_emb = ggml_add(ctx0, pos_emb, ggml_mul(ctx0, ggml_get_rows(ctx0, model.position_embeddings, pos_idx2), pos_w2));
    pos_emb = ggml_add(ctx0, pos_emb, ggml_mul(ctx0, ggml_get_rows(ctx0, model.position_embeddings, pos_idx3), pos_w3));

    patch_emb = ggml_add(ctx0, patch_emb, pos_emb);

    // 3. Prepend `CLS = v.class_embd + v.cls_pos_embd` at index 0 -> `[1024, 1 + n_tok, 1]`
    ggml_tensor * cls_tok = ggml_add(ctx0, model.class_embedding, model.cls_pos_embedding);
    cls_tok = ggml_reshape_2d(ctx0, cls_tok, n_embd, 1);
    ggml_tensor * vit_inp = ggml_concat(ctx0, cls_tok, patch_emb, 1);
    vit_inp = ggml_reshape_3d(ctx0, vit_inp, n_embd, n_seq, 1);

    // 4. 2D Axial RoPE (`GGML_ROPE_TYPE_NORM`: dims `[0..31]` = `wpos + 1`, dims `[32..63]` = `hpos + 1`, CLS = `(0, 0)`)
    ggml_tensor * rope_w = inp_i32("ink2_rope_w", n_seq);
    ggml_tensor * rope_h = inp_i32("ink2_rope_h", n_seq);

    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        return build_rope_2d(ctx0, cur, rope_w, rope_h, hparams.rope_theta, false);
    };

    ggml_tensor * vit_out = build_vit(
            vit_inp, n_seq,
            NORM_TYPE_NORMAL,
            FFN_GELU_ERF,
            nullptr,
            add_pos);

    // 5. Strip CLS token 0 -> 2x2 merge `[4096, n_out]` -> `mm.0 -> GELU_ERF -> mm.2` -> `[2304, n_out]`
    ggml_tensor * patches_out = ggml_view_2d(ctx0, vit_out, n_embd, n_tok, vit_out->nb[1], vit_out->nb[1]);
    patches_out = ggml_cont_2d(ctx0, patches_out, n_embd * merge * merge, n_out);

    ggml_tensor * emb = build_ffn(
            patches_out,
            model.mm_0_w, model.mm_0_b,
            nullptr, nullptr,
            model.mm_2_w, model.mm_2_b,
            FFN_GELU_ERF,
            -1);

    ggml_build_forward_expand(gf, emb);
    return gf;
}
