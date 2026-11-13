/* Minimal C stand-in for libps6000, to test the ctypes calls of picodaq.ps6000
 * against a real shared library (argument types, pointers, struct layout).
 * Every capture is "ready" at once; segment i of channel ch holds 100*ch + i. */
#include <stdint.h>
#include <string.h>

typedef uint32_t PICO_STATUS;
#pragma pack(push, 1)
typedef struct {
    uint32_t status, segmentIndex, triggerIndex;
    uint64_t triggerTime;
    uint16_t timeUnits, reserved0;
    uint64_t timeStampCounter;
} PS6000_TRIGGER_INFO;
#pragma pack(pop)

static int16_t *bufs[4][64];
static uint32_t nseg = 1, ncap = 1, nsamp = 0;
static float offsets[4];
int16_t last_threshold = 0;
int32_t last_range[4];

PICO_STATUS ps6000OpenUnit(int16_t *h, char *serial) { *h = 3; return 0; }
PICO_STATUS ps6000CloseUnit(int16_t h) { return 0; }
PICO_STATUS ps6000GetUnitInfo(int16_t h, char *s, int16_t n, int16_t *req, uint32_t info) {
    const char *t = info == 3 ? "6402C" : info == 4 ? "SHIM/0001" : "";
    strncpy(s, t, n); *req = (int16_t)strlen(t); return 0;
}
PICO_STATUS ps6000SetChannel(int16_t h, int32_t ch, int16_t en, int32_t cpl, int32_t rng, float off, int32_t bw) {
    if (ch < 0 || ch > 3) return 0x10;
    offsets[ch] = off; last_range[ch] = rng; return 0;
}
PICO_STATUS ps6000GetAnalogueOffset(int16_t h, int32_t rng, int32_t cpl, float *mx, float *mn) {
    *mx = 0.5f; *mn = -0.5f; return 0;
}
PICO_STATUS ps6000GetTimebase2(int16_t h, uint32_t tb, uint32_t n, float *dt, int16_t os, uint32_t *mx, uint32_t seg) {
    *dt = tb < 5 ? (float)(1u << tb) / 5.0f : (float)(tb - 4) / 0.15625f; *mx = 1u << 28; return 0;
}
PICO_STATUS ps6000SetSimpleTrigger(int16_t h, int16_t en, int32_t ch, int16_t thr, int32_t dir, uint32_t d, int16_t a) {
    last_threshold = thr; return 0;
}
PICO_STATUS ps6000MemorySegments(int16_t h, uint32_t n, uint32_t *mx) { nseg = n; *mx = (1u << 28) / n; return 0; }
PICO_STATUS ps6000SetNoOfCaptures(int16_t h, uint32_t n) { if (n > nseg || n > 64) return 0x42; ncap = n; return 0; }
PICO_STATUS ps6000SetDataBufferBulk(int16_t h, int32_t ch, int16_t *b, uint32_t len, uint32_t seg, int32_t m) {
    if (seg >= 64) return 0x26;
    bufs[ch][seg] = b; nsamp = len; return 0;
}
PICO_STATUS ps6000RunBlock(int16_t h, uint32_t pre, uint32_t post, uint32_t tb, int16_t os, int32_t *ti,
                           uint32_t seg, void *cb, void *p) { return 0; }
PICO_STATUS ps6000IsReady(int16_t h, int16_t *r) { *r = 1; return 0; }
PICO_STATUS ps6000Stop(int16_t h) { return 0; }
PICO_STATUS ps6000GetNoOfCaptures(int16_t h, uint32_t *n) { *n = ncap; return 0; }
PICO_STATUS ps6000GetValuesBulk(int16_t h, uint32_t *n, uint32_t from, uint32_t to, uint32_t r, int32_t m, int16_t *ov) {
    for (int ch = 0; ch < 4; ch++)
        for (uint32_t s = from; s <= to; s++)
            if (bufs[ch][s]) for (uint32_t i = 0; i < nsamp; i++) bufs[ch][s][i] = (int16_t)(100 * ch + s);
    for (uint32_t s = from; s <= to; s++) ov[s - from] = 0;
    *n = nsamp; return 0;
}
PICO_STATUS ps6000GetTriggerInfoBulk(int16_t h, PS6000_TRIGGER_INFO *info, uint32_t from, uint32_t to) {
    for (uint32_t s = from; s <= to; s++) {
        memset(&info[s - from], 0, sizeof(PS6000_TRIGGER_INFO));
        info[s - from].segmentIndex = s;
        info[s - from].timeStampCounter = 0x00FFFFFFFFFFFF00ull + 1000ull * s;  /* wraps at 2^56 */
    }
    return 0;
}
