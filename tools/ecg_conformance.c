/*
 * ecg_conformance - does an engine library keep the standard interface?
 *
 *   ecg_conformance path/to/libecg.{so,dylib}
 *
 * A host that swaps engines runs this against the new file before it trusts
 * it. It is also the smallest example of such a host: it knows the engine only
 * by the path it was given and the functions ecg.h declares, loads it at run
 * time, and checks the ABI version before calling anything else.
 *
 * What it checks is the contract, not accuracy - that is the evaluation
 * harness's job. Exit status 0 means every check passed.
 */
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ecg.h"

static int failures = 0;
#define CHECK(cond, ...)                                                                           \
    do {                                                                                           \
        if (cond) {                                                                                \
            printf("  ok    ");                                                                    \
        } else {                                                                                   \
            printf("  FAIL  ");                                                                    \
            failures++;                                                                            \
        }                                                                                          \
        printf(__VA_ARGS__);                                                                       \
        printf("\n");                                                                              \
    } while (0)

static struct {
    ecg_abi_version_fn abi_version;
    ecg_engine_id_fn engine_id;
    ecg_channel_create_fn create;
    ecg_channel_destroy_fn destroy;
    ecg_channel_push_fn push;
    ecg_channel_gap_fn gap;
    ecg_channel_finish_fn finish;
    ecg_channel_poll_fn poll;
    ecg_channel_status_fn status;
    /* 1.1; null when the engine predates them */
    ecg_engine_stages_fn engine_stages;
    ecg_channel_stages_fn channel_stages;
} E;

static int load(const char *path) {
    void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        printf("cannot load %s: %s\n", path, dlerror());
        return 0;
    }
#define SYM(field, name)                                                                           \
    do {                                                                                           \
        *(void **)(&E.field) = dlsym(h, name);                                                     \
        if (!E.field) {                                                                            \
            printf("missing symbol %s\n", name);                                                   \
            return 0;                                                                              \
        }                                                                                          \
    } while (0)
    SYM(abi_version, "ecg_abi_version");
    SYM(engine_id, "ecg_engine_id");
    SYM(create, "ecg_channel_create");
    SYM(destroy, "ecg_channel_destroy");
    SYM(push, "ecg_channel_push");
    SYM(gap, "ecg_channel_gap");
    SYM(finish, "ecg_channel_finish");
    SYM(poll, "ecg_channel_poll");
    SYM(status, "ecg_channel_status");
    *(void **)(&E.engine_stages) = dlsym(h, "ecg_engine_stages");
    *(void **)(&E.channel_stages) = dlsym(h, "ecg_channel_stages");
    return 1;
}

/* A synthetic ECG: narrow complexes at a fixed rate on a quiet baseline. */
static float synth(size_t i, double fs, double bpm) {
    double period = 60.0 / bpm;
    double t = fmod((double)i / fs, period);
    double qrs = exp(-pow((t - 0.25) / 0.012, 2.0));
    double tw = 0.25 * exp(-pow((t - 0.55) / 0.06, 2.0));
    return (float)(qrs + tw);
}

typedef struct {
    size_t beats, unknown_kinds, unknown_codes, bad_spans, bad_scores, unordered;
    uint64_t last_beat;
    int any_beat;
} tally;

static void account(tally *t, const ecg_event *ev, int64_t n) {
    for (int64_t k = 0; k < n; k++) {
        const ecg_event *e = &ev[k];
        if (e->end < e->start)
            t->bad_spans++;
        switch (e->kind) {
        case ECG_EV_BEAT:
            t->beats++;
            if (e->code > ECG_BEAT_UNKNOWN)
                t->unknown_codes++;
            for (int j = 0; j < 3; j++)
                if (!(e->score[j] >= 0.0f && e->score[j] <= 1.0f))
                    t->bad_scores++;
            if (t->any_beat && e->start <= t->last_beat)
                t->unordered++;
            t->last_beat = e->start;
            t->any_beat = 1;
            break;
        case ECG_EV_RHYTHM:
        case ECG_EV_AF_WINDOW:
        case ECG_EV_VF:
        case ECG_EV_LEAD_OFF:
        case ECG_EV_SV_RUN:
            break;
        default:
            /* Allowed: a newer engine may report kinds this header predates. */
            t->unknown_kinds++;
        }
    }
}

/* Push `seconds` of signal in uneven chunks, polling through a small buffer
 * after each push so partial drains are exercised. */
static void run(ecg_channel *ch, tally *t, double fs, double bpm, double seconds, size_t offset) {
    size_t total = (size_t)(fs * seconds), i = 0, step = 37;
    float buf[1024];
    ecg_event ev[7];
    while (i < total) {
        size_t n = step < total - i ? step : total - i;
        for (size_t k = 0; k < n; k++)
            buf[k] = synth(offset + i + k, fs, bpm);
        if (E.push(ch, buf, n) != ECG_OK)
            return;
        int64_t got;
        while ((got = E.poll(ch, ev, 7)) > 0)
            account(t, ev, got);
        i += n;
        step = step * 7 % 1000 + 1; /* 1..1000, uneven */
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s path/to/libecg\n", argv[0]);
        return 2;
    }
    if (!load(argv[1]))
        return 1;

    uint32_t v = E.abi_version();
    printf("engine: %s\nabi:    %u.%u (host expects %d.x)\n", E.engine_id(), v >> 16, v & 0xffff,
           ECG_ABI_MAJOR);
    CHECK((v >> 16) == ECG_ABI_MAJOR, "ABI major matches");
    if ((v >> 16) != ECG_ABI_MAJOR)
        return 1;

    int32_t err = 0;
    ecg_config bad = {sizeof(ecg_config), 99, 250.0, NULL};
    CHECK(E.create(&bad, &err) == NULL && err == ECG_ERR_CONFIG, "unknown preset refused");
    ecg_config nofs = {sizeof(ecg_config), ECG_PRESET_CLINICAL, 0.0, NULL};
    CHECK(E.create(&nofs, &err) == NULL && err == ECG_ERR_CONFIG, "zero sampling rate refused");
    ecg_config shortcfg = {4, ECG_PRESET_CLINICAL, 250.0, NULL};
    CHECK(E.create(&shortcfg, &err) == NULL && err == ECG_ERR_CONFIG, "truncated config refused");
    CHECK(E.create(NULL, &err) == NULL && err == ECG_ERR_NULL, "null config refused");

    const double fs = 250.0, bpm = 72.0, seconds = 120.0;
    ecg_config cfg = {sizeof(ecg_config), ECG_PRESET_CLINICAL, fs, NULL};
    ecg_channel *a = E.create(&cfg, &err);
    ecg_channel *b = E.create(&cfg, &err);
    CHECK(a && b && err == ECG_OK, "two clinical channels created");
    if (!a || !b)
        return 1;

    tally ta = {0}, tb = {0};
    run(a, &ta, fs, bpm, seconds, 0);
    run(b, &tb, fs, bpm, seconds, 0);
    double expected = bpm * seconds / 60.0;
    CHECK(ta.beats >= 0.8 * expected && ta.beats <= 1.05 * expected,
          "beats reported: %zu of about %.0f", ta.beats, expected);
    CHECK(ta.unordered == 0, "beats arrive in time order");
    CHECK(ta.bad_spans == 0, "every event ends at or after it starts");
    CHECK(ta.bad_scores == 0, "beat scores are probabilities");
    CHECK(ta.beats == tb.beats && ta.last_beat == tb.last_beat,
          "two channels given the same signal agree (%zu, %zu)", ta.beats, tb.beats);
    if (ta.unknown_kinds || ta.unknown_codes)
        printf("  note  %zu events of kinds and %zu of codes this host does not know; skipped\n",
               ta.unknown_kinds, ta.unknown_codes);

    ecg_status st;
    memset(&st, 0, sizeof st);
    st.struct_size = sizeof st;
    CHECK(E.status(a, &st) == ECG_OK, "status read");
    CHECK(st.struct_size <= sizeof st && st.struct_size >= 24, "status size written back: %u",
          st.struct_size);
    CHECK(st.samples == (uint64_t)(fs * seconds), "status counts the samples pushed: %llu",
          (unsigned long long)st.samples);
    CHECK(st.quality <= ECG_QUALITY_UNKNOWN, "status quality is a known level");

    CHECK(E.gap(a, 500) == ECG_OK, "gap accepted");
    run(a, &ta, fs, bpm, 10.0, (size_t)(fs * seconds) + 500);
    E.status(a, &st);
    CHECK(st.samples == (uint64_t)(fs * seconds) + 500 + (uint64_t)(fs * 10.0),
          "the time base advances through a gap");
    CHECK(ta.unordered == 0, "beats stay in order across the gap");

    CHECK(E.finish(a) == ECG_OK, "finish");
    ecg_event ev[64];
    int64_t n;
    while ((n = E.poll(a, ev, 64)) > 0)
        account(&ta, ev, n);
    CHECK(n == 0, "poll drains to zero");

    CHECK(E.push(NULL, NULL, 0) == ECG_ERR_NULL, "null channel refused");
    CHECK(E.push(a, NULL, 5) == ECG_ERR_NULL, "null samples refused");
    CHECK(E.poll(a, NULL, 5) == ECG_ERR_NULL, "null poll buffer refused");

    E.destroy(a);
    E.destroy(b);
    E.destroy(NULL);
    printf("  ok    destroy, including null\n");

    ecg_config patch = {sizeof(ecg_config), ECG_PRESET_PATCH, 250.0, NULL};
    ecg_channel *p = E.create(&patch, &err);
    CHECK(p != NULL && err == ECG_OK, "patch preset created");
    E.destroy(p);

    /* A host written against 1.0 passes a config that ends after fs. */
    ecg_config v10 = {16, ECG_PRESET_CLINICAL, 250.0, NULL};
    ecg_channel *old = E.create(&v10, &err);
    CHECK(old != NULL && err == ECG_OK, "a 1.0-sized config is still accepted");
    E.destroy(old);

    if ((v & 0xffff) >= 1) {
        CHECK(E.engine_stages && E.channel_stages, "1.1 stage functions present");
        if (E.engine_stages && E.channel_stages) {
            const char *list = E.engine_stages();
            size_t n_stages = 0, n_ok = 0;
            char line[256];
            const char *q = list;
            while (*q) {
                /* name is up to the tab; kind is up to the first dot */
                const char *tab = strchr(q, '\t');
                const char *nl = strchr(q, '\n');
                if (!tab || (nl && nl < tab))
                    break;
                size_t len = (size_t)(tab - q);
                const char *dot = memchr(q, '.', len);
                if (dot && len < 100) {
                    char name[128], kind[32], spec[192];
                    memcpy(name, q, len);
                    name[len] = 0;
                    size_t kl = (size_t)(dot - q) < 31 ? (size_t)(dot - q) : 31;
                    memcpy(kind, q, kl);
                    kind[kl] = 0;
                    snprintf(spec, sizeof spec, "%s=%s", kind, name);
                    ecg_config c = {sizeof(ecg_config), ECG_PRESET_CLINICAL, 250.0, spec};
                    ecg_channel *s = E.create(&c, &err);
                    n_stages++;
                    if (s) {
                        tally t = {0};
                        run(s, &t, 250.0, 72.0, 30.0, 0);
                        const char *ids = E.channel_stages(s);
                        snprintf(line, sizeof line, "%s", ids ? ids : "");
                        if (ids && strstr(ids, name) && t.unordered == 0 && t.bad_spans == 0 &&
                            t.beats > 20)
                            n_ok++;
                        else
                            printf("  FAIL  stage %s: runs as \"%s\", %zu beats\n", name, line,
                                   t.beats);
                        E.destroy(s);
                    } else {
                        printf("  FAIL  stage %s refused (%d)\n", name, err);
                    }
                }
                q = nl ? nl + 1 : q + strlen(q);
            }
            CHECK(n_stages > 0 && n_ok == n_stages,
                  "every listed stage can be selected and runs (%zu of %zu)", n_ok, n_stages);
            ecg_config bogus = {sizeof(ecg_config), ECG_PRESET_CLINICAL, 250.0,
                                "vf=vf.does-not-exist@1"};
            CHECK(E.create(&bogus, &err) == NULL && err == ECG_ERR_CONFIG,
                  "an unknown stage is refused, not replaced");
            ecg_config def = {sizeof(ecg_config), ECG_PRESET_CLINICAL, 250.0, NULL};
            ecg_channel *d = E.create(&def, &err);
            printf("  info  default stages: %s\n", d ? E.channel_stages(d) : "?");
            E.destroy(d);
        }
    }

    printf("%s: %d check(s) failed\n", failures ? "NOT CONFORMANT" : "CONFORMANT", failures);
    return failures ? 1 : 0;
}
