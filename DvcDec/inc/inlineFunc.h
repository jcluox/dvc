#ifndef INLINEFUNC_H
#define INLINEFUNC_H

static inline int imin(int a, int b){
    return ((a) < (b)) ? (a) : (b);
}

static inline int imax(int a, int b)
{
    return ((a) > (b)) ? (a) : (b);
}

static inline int iabs(int x){
    return ((x) < 0) ? -(x) : (x);
}

static inline int isignab(int a, int b)
{
    return ((b) < 0) ? -iabs(a) : iabs(a);
}

static inline int iClip1(int high, int x)
{
    x = imax(x, 0);
    x = imin(x, high);

    return x;
}

static inline int iClip3(int low, int high, int x)
{
    x = imax(x, low);
    x = imin(x, high);

    return x;
}

static inline int rshift_rnd_sf(int x, int a)
{
    return ((x + (1 << (a-1) )) >> a);
}


#endif
