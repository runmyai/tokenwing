#include "../src/unicode.h"

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

int main() {
    bool success = true;
    const struct {
        std::string text;
        std::vector<uint32_t> expected;
    } utf8_tests[] = {
        { "\x7f",                 { 0x7f } },
        { "\xc2\x80",             { 0x80 } },
        { "\xdf\xbf",             { 0x7ff } },
        { "\xe0\xa0\x80",         { 0x800 } },
        { "\xef\xbf\xbf",         { 0xffff } },
        { "\xf0\x90\x80\x80",     { 0x10000 } },
        { "\xf4\x8f\xbf\xbf",     { 0x10ffff } },
        // Invalid-input cases preserve the existing recovery behavior: one U+FFFD per byte.
        { "\xc0\xa0",             { 0xfffd, 0xfffd } },
        { "\xc1\xbf",             { 0xfffd, 0xfffd } },
        { "\xe0\x80\xa0",         { 0xfffd, 0xfffd, 0xfffd } },
        { "\xe0\x9f\xbf",         { 0xfffd, 0xfffd, 0xfffd } },
        { "\xf0\x80\x80\xa0",     { 0xfffd, 0xfffd, 0xfffd, 0xfffd } },
        { "\xf0\x8f\xbf\xbf",     { 0xfffd, 0xfffd, 0xfffd, 0xfffd } },
        { "\xf4\x90\x80\x80",     { 0xfffd, 0xfffd, 0xfffd, 0xfffd } },
        { "\xf5\x80\x80\x80",     { 0xfffd, 0xfffd, 0xfffd, 0xfffd } },
        { "\xf7\xbf\xbf\xbf",     { 0xfffd, 0xfffd, 0xfffd, 0xfffd } },
        { "\xe2\x80",             { 0xfffd, 0xfffd } },
    };

    for (const auto & test : utf8_tests) {
        auto expected = test.expected;
        expected.insert(expected.begin(), 'a');
        expected.push_back('b');
        if (unicode_cpts_from_utf8("a" + test.text + "b") != expected) {
            fprintf(stderr, "unexpected UTF-8 decoding for bytes:");
            for (const unsigned char byte : test.text) {
                fprintf(stderr, " %02x", byte);
            }
            fprintf(stderr, "\n");
            success = false;
        }
    }

    const std::vector<std::string> regex_exprs = {
        "[~][A-Za-z]+| ?[\\p{S}]+|\\s+",
    };
    const std::vector<std::string> expected = { " ~", "foo" };
    const auto actual = unicode_regex_split(" ~foo", regex_exprs, false);

    if (actual != expected) {
        fprintf(stderr, "unexpected split:");
        for (const auto & piece : actual) {
            fprintf(stderr, " [%s]", piece.c_str());
        }
        fprintf(stderr, "\n");
        success = false;
    }

    // The contraction-aware word splitter from issue #23 (GPT-2/tiktoken shape) used to throw
    // regex_error(error_complexity) on MSVC during matching, failing the whole request. It must now
    // fall back to splitting on whitespace and keep every codepoint of the text.
    const std::vector<std::string> word_exprs = {
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    };
    const std::string long_text = "Hello, world!  foo_bar\t123\nend. (x) [y] {z} 's 'll  mixedCASE caf\xc3\xa9 \xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e\n";
    const auto words = unicode_regex_split(long_text, word_exprs, false);
    std::string rejoined;
    for (const auto & word : words) {
        rejoined += word;
    }
    if (rejoined != long_text) {
        fprintf(stderr, "word splitter did not preserve the text (%zu pieces, %zu in, %zu out)\n",
                words.size(), long_text.size(), rejoined.size());
        success = false;
    }

    // Flint's word split has a hand-written splitter (std::regex gives up on long input on MSVC). It must cut exactly where the
    // regex path does: the same pattern inside (?:...) is not recognised as Flint's and runs through std::regex, so both are
    // compared on edge cases, on random strings and on a source file, alone and as the third of Flint's three patterns.
    {
        const std::vector<std::string> flint = { "\\p{N}{1,3}", "[一-龥぀-ゟ゠-ヿ]+", "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" };
        const std::vector<std::string> flint_regex = { flint[0], flint[1], "(?:" + flint[2] + ")" };
        const char * pieces[] = {
            "a", "Z", "q", "\xc3\xa9", "\xc3\x89", "\xd0\x96", "\xd0\xb6", "\xe6\x97\xa5", "\xe3\x81\x82", "\xcc\x81", "0", "7", "\xd9\xa3",
            " ", "  ", "\t", "\n", "\r", "\r\n", "\xc2\xa0", "\xe3\x80\x80", "\x0b", "\x0c", "\x01", "\x7f", "\xee\x80\x80",
            "'", "'s", "'S", "'t", "'re", "'RE", "'ve", "'m", "'ll", "'Ll", "'d", "'x", ".", ",", "!", "_", "-", "(", ")", "{", "}",
            "$", "+", "<", "=", "^", "`", "|", "~", "\xe2\x80\x9c", "\xe2\x80\x94", "\xef\xbc\x8c", "\xe2\x82\xac", "\xc2\xa9",
            "\xf0\x9f\x98\x80", "Hello", "world", "camelCase", "SNAKE_CASE", "don", "it",
        };
        std::vector<std::string> texts = {
            "", "Hello, world!  foo_bar\t123\nend. (x) [y] {z} 's 'll  mixedCASE caf\xc3\xa9 \xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e\n",
            "don't won'T we're THEY'VE i'm we'll he'd it'x 'tis", "  \n\n  x  \r\n\t y\t\t", "a\xcc\x81b\xcc\x81 \xcc\x81\xcc\x81 x",
            " !!\n\n x ,, \xe2\x80\x9cq\xe2\x80\x9d\r\r", "\x01\x01a \x7f\x7f \xee\x80\x80\xee\x80\x80b 12 345",
        };
        uint32_t seed = 12345;
        const auto rnd = [&seed]() { seed = seed * 1103515245u + 12345u; return (seed >> 16) & 0x7fff; };
        for (int t = 0; t < 20000; t++) {
            std::string s;
            const int n = rnd() % 24;
            for (int i = 0; i < n; i++) {
                s += pieces[rnd() % (sizeof(pieces) / sizeof(pieces[0]))];
            }
            texts.push_back(s);
        }
        {
            FILE * f = fopen(__FILE__, "rb");   // a real source file: code, comments, escapes
            if (f) {
                std::string src;
                char buf[4096];
                size_t got;
                while ((got = fread(buf, 1, sizeof(buf), f)) > 0) {
                    src.append(buf, got);
                }
                fclose(f);
                texts.push_back(src);
            }
        }
        size_t bad = 0;
        for (const auto & text : texts) {
            const bool same3 = unicode_regex_split(text, { flint[2] }, false) == unicode_regex_split(text, { flint_regex[2] }, false);
            const bool same_all = unicode_regex_split(text, flint, false) == unicode_regex_split(text, flint_regex, false);
            if (!same3 || !same_all) {
                if (bad++ < 5) {
                    fprintf(stderr, "Flint's word splitter differs from the regex path on: '%s'\n", text.c_str());
                }
            }
        }
        if (bad > 0) {
            fprintf(stderr, "Flint's word splitter differs from the regex path on %zu of %zu texts\n", bad, texts.size());
            success = false;
        }
    }

    return success ? 0 : 1;
}
