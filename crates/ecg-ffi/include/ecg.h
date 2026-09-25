/*
 * ecg.h - the live-ecg engine's standard interface, ABI 1.0.
 *
 * The engine is meant to be replaced often; this boundary is not. A host
 * written against this header runs any engine whose ecg_abi_version() has the
 * same major number, whether it links the library or loads it at run time.
 *
 * Compatibility rules
 *   - ecg_abi_version() returns (major << 16) | minor. Refuse a different major.
 *   - Within a major an engine only adds: new event kinds, new codes, new
 *     fields at the END of ecg_config and ecg_status. Nothing is renumbered
 *     or removed.
 *   - Skip any event kind or code you do not know. A newer engine may report
 *     things this header does not list.
 *   - Set struct_size on every struct you pass in. The engine reads and writes
 *     no further than it says.
 *   - Nothing unwinds across this boundary. An internal failure returns
 *     ECG_ERR_INTERNAL, and that channel then returns ECG_ERR_POISONED; destroy
 *     it and create a new one.
 *
 * Use
 *   One channel per signal. A channel is not shared between threads; separate
 *   channels are independent and may run on different threads. Samples are in
 *   millivolts. Poll after each push: events queue inside the channel until
 *   they are polled. Sample indices count from zero at the first sample pushed
 *   and advance through gaps.
 */
#ifndef LIVE_ECG_H
#define LIVE_ECG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ECG_ABI_MAJOR 1
#define ECG_ABI_MINOR 0

/* results */
#define ECG_OK            0
#define ECG_ERR_NULL     -1
#define ECG_ERR_CONFIG   -2
#define ECG_ERR_INTERNAL -3
#define ECG_ERR_POISONED -4

/* presets */
#define ECG_PRESET_CLINICAL 0 /* clinical electrodes; the default */
#define ECG_PRESET_PATCH    1 /* single-lead patch worn for days */

/* event kinds and their codes */
#define ECG_EV_BEAT 1 /* start == end == R peak; score = p(V), p(S), p(F); aux = morphology */
#define   ECG_BEAT_N       0
#define   ECG_BEAT_S       1
#define   ECG_BEAT_V       2
#define   ECG_BEAT_F       3
#define   ECG_BEAT_UNKNOWN 4 /* not judged: signal quality, or no template yet */
#define   ECG_BEAT_FLAG_FIBRILLATING 1u
#define ECG_EV_RHYTHM 2 /* an episode that has ended; code = condition */
#define   ECG_RHYTHM_PAUSE                    1
#define   ECG_RHYTHM_ASYSTOLE                 2
#define   ECG_RHYTHM_BRADYCARDIA              3
#define   ECG_RHYTHM_TACHYCARDIA              4
#define   ECG_RHYTHM_VENTRICULAR_RUN          5
#define   ECG_RHYTHM_VENTRICULAR_TACHYCARDIA  6
#define   ECG_RHYTHM_BIGEMINY                 7
#define   ECG_RHYTHM_TRIGEMINY                8
#define   ECG_RHYTHM_IDIOVENTRICULAR          9
#define ECG_EV_AF_WINDOW 3 /* one AF decision window; score[0] = probability */
#define   ECG_AF_FLAG_IN_AF 1u
#define ECG_EV_VF 4        /* a ventricular fibrillation episode that has ended */
#define ECG_EV_LEAD_OFF 5  /* an electrode failure that has ended; code = kind */
#define   ECG_LEAD_OFF_RAIL 1
#define   ECG_LEAD_OFF_OPEN 2
#define ECG_EV_SV_RUN 6    /* a supraventricular run that has ended; aux = beats.
                              Report an N beat inside it as S. */

/* status */
#define ECG_QUALITY_GOOD       0
#define ECG_QUALITY_ACCEPTABLE 1
#define ECG_QUALITY_UNUSABLE   2
#define ECG_QUALITY_UNKNOWN    3
#define ECG_STATE_IN_AF       1u
#define ECG_STATE_IN_VF       2u
#define ECG_STATE_LEAD_OFF    4u
#define ECG_STATE_SUPPRESSING 8u /* beat findings withheld: fibrillation suspected */

typedef struct ecg_config {
    uint32_t struct_size; /* sizeof(ecg_config) */
    uint32_t preset;
    double fs;            /* Hz */
} ecg_config;

typedef struct ecg_event {
    uint32_t kind;
    uint32_t code;
    uint32_t flags;
    uint32_t aux;
    uint64_t start;
    uint64_t end;
    float score[4];
} ecg_event;

typedef struct ecg_status {
    uint32_t struct_size; /* set to sizeof(ecg_status); the engine writes back what it filled */
    uint32_t quality;
    uint32_t state;
    uint32_t reserved;
    uint64_t samples;
    float quality_score;  /* 0..1; negative before the first window */
} ecg_status;

typedef struct ecg_channel ecg_channel;

uint32_t ecg_abi_version(void);
const char *ecg_engine_id(void);

ecg_channel *ecg_channel_create(const ecg_config *cfg, int32_t *err);
void ecg_channel_destroy(ecg_channel *ch);

int32_t ecg_channel_push(ecg_channel *ch, const float *samples_mv, size_t n);
int32_t ecg_channel_gap(ecg_channel *ch, uint64_t samples);
int32_t ecg_channel_finish(ecg_channel *ch);

/* Returns the number of events copied (0..cap), or a negative error code. */
int64_t ecg_channel_poll(ecg_channel *ch, ecg_event *out, size_t cap);
int32_t ecg_channel_status(ecg_channel *ch, ecg_status *out);

/* Function-pointer types, for hosts that load the engine at run time. */
typedef uint32_t (*ecg_abi_version_fn)(void);
typedef const char *(*ecg_engine_id_fn)(void);
typedef ecg_channel *(*ecg_channel_create_fn)(const ecg_config *, int32_t *);
typedef void (*ecg_channel_destroy_fn)(ecg_channel *);
typedef int32_t (*ecg_channel_push_fn)(ecg_channel *, const float *, size_t);
typedef int32_t (*ecg_channel_gap_fn)(ecg_channel *, uint64_t);
typedef int32_t (*ecg_channel_finish_fn)(ecg_channel *);
typedef int64_t (*ecg_channel_poll_fn)(ecg_channel *, ecg_event *, size_t);
typedef int32_t (*ecg_channel_status_fn)(ecg_channel *, ecg_status *);

#ifdef __cplusplus
}
#endif
#endif /* LIVE_ECG_H */
