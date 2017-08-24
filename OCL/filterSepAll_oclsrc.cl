#define RADIUSX 2
#define LSIZE0 16
#define LSIZE1 16
#define CN 1
#define BORDER_REFLECT_101
#define NO_EXTRA_EXTRAPOLATION
#define NO_BORDER_ISOLATED
#define srcTrow uchar
#define dstTrow int
#define convertToDstTrow convert_int
#define srcT1row uchar
#define dstT1row int
#define DOUBLE_SUPPORT
#define INTEGER_ARITHMETIC


#define RADIUSY 2
#define srcTcol int
#define dstTcol uchar
#define convertToDstTcol convert_uchar_sat
#define srcT1col int
#define dstT1col uchar
#define SHIFT_BITS 16


#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif


#define READ_TIMES_ROW ((2*(RADIUSX+LSIZE0)-1)/LSIZE0)
#define RADIUS 1
#ifdef BORDER_REPLICATE
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? (l_edge)   : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? (r_edge)-1 : (addr))
#endif
#ifdef BORDER_REFLECT
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? -(i)-1               : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? -(i)-1+((r_edge)<<1) : (addr))
#endif
#ifdef BORDER_REFLECT_101
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? -(i)                 : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? -(i)-2+((r_edge)<<1) : (addr))
#endif
#ifdef BORDER_WRAP
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? (i)+(r_edge) : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? (i)-(r_edge) : (addr))
#endif
#ifdef EXTRA_EXTRAPOLATION
#ifdef BORDER_CONSTANT
#define ELEM(i,l_edge,r_edge,elem1,elem2) (i)<(l_edge) | (i) >= (r_edge) ? (elem1) : (elem2)
#elif defined BORDER_REPLICATE
#define EXTRAPOLATE(t, minT, maxT) \
{ \
t = max(min(t, (maxT) - 1), (minT)); \
}
#elif defined BORDER_WRAP
#define EXTRAPOLATE(x, minT, maxT) \
{ \
if (t < (minT)) \
t -= ((t - (maxT) + 1) / (maxT)) * (maxT); \
if (t >= (maxT)) \
t %= (maxT); \
}
#elif defined(BORDER_REFLECT) || defined(BORDER_REFLECT_101)
#define EXTRAPOLATE_(t, minT, maxT, delta) \
{ \
if ((maxT) - (minT) == 1) \
t = (minT); \
else \
do \
{ \
if (t < (minT)) \
t = (minT) - (t - (minT)) - 1 + delta; \
else \
t = (maxT) - 1 - (t - (maxT)) - delta; \
} \
while (t >= (maxT) || t < (minT)); \
\
}
#ifdef BORDER_REFLECT
#define EXTRAPOLATE(t, minT, maxT) EXTRAPOLATE_(t, minT, maxT, 0)
#elif defined(BORDER_REFLECT_101)
#define EXTRAPOLATE(t, minT, maxT) EXTRAPOLATE_(t, minT, maxT, 1)
#endif
#else
#error No extrapolation method
#endif
#else
#ifdef BORDER_CONSTANT
#define ELEM(i,l_edge,r_edge,elem1,elem2) (i)<(l_edge) | (i) >= (r_edge) ? (elem1) : (elem2)
#else
#define EXTRAPOLATE(t, minT, maxT) \
{ \
int _delta = t - (minT); \
_delta = ADDR_L(_delta, 0, (maxT) - (minT)); \
_delta = ADDR_R(_delta, (maxT) - (minT), _delta); \
t = _delta + (minT); \
}
#endif
#endif


#define READ_TIMES_COL ((2*(RADIUSY+LSIZE1)-1)/LSIZE1)


#define noconvert


#if CN != 3
#define loadpixrow(addr) *(__global const srcTrow *)(addr)
#define storepixrow(val, addr)  *(__global dstTrow *)(addr) = val
#define SRCSIZErow (int)sizeof(srcTrow)
#define DSTSIZErow (int)sizeof(dstTrow)
#else
#define loadpixrow(addr)  vload3(0, (__global const srcT1row *)(addr))
#define storepixrow(val, addr) vstore3(val, 0, (__global dstT1row *)(addr))
#define SRCSIZErow (int)sizeof(srcT1row)*3
#define DSTSIZErow (int)sizeof(dstT1row)*3
#endif


#if CN != 3
#define loadpixcol(addr) *(__global const srcTcol *)(addr)
#define storepixcol(val, addr)  *(__global dstTcol *)(addr) = val
#define SRCSIZEcol (int)sizeof(srcTcol)
#define DSTSIZEcol (int)sizeof(dstTcol)
#else
#define loadpixcol(addr)  vload3(0, (__global const srcT1col *)(addr))
#define storepixcol(val, addr) vstore3(val, 0, (__global dstT1col *)(addr))
#define SRCSIZEcol (int)sizeof(srcT1col)*3
#define DSTSIZEcol (int)sizeof(dstT1col)*3
#endif


#define DIG(a) a,

#define COEFF DIG(31)DIG(60)DIG(75)DIG(60)DIG(31)



// ~/opencv-3.2.0/builds/modules/imgproc/opencl_kernels_imgproc.cpp
//OPENCL SEP ROW FILTER 2D KERNEL:

__constant dstT1row mat_kernelrow[] = { COEFF };
#if (defined(INTEGER_ARITHMETIC) && !INTEL_DEVICE)
#define dstT4 int4
#define convertDstVec convert_int4
#else
#define dstT4 float4
#define convertDstVec convert_float4
#endif

__kernel void row_filter_C1_D0(__global const uchar * src, int src_step_in_pixel, int src_offset_x, int src_offset_y,
int src_cols, int src_rows, int src_whole_cols, int src_whole_rows,
        __global float * dst, int dst_step_in_pixel, int dst_cols, int dst_rows,
int radiusy)
{
int x = get_global_id(0)<<2;
int y = get_global_id(1);
int l_x = get_local_id(0);
int l_y = get_local_id(1);
int start_x = x + src_offset_x - RADIUSX & 0xfffffffc;
int offset = src_offset_x - RADIUSX & 3;
int start_y = y + src_offset_y - radiusy;
int start_addr = mad24(start_y, src_step_in_pixel, start_x);
dstT4 sum;
uchar4 temp[READ_TIMES_ROW];
__local uchar4 LDS_DAT[LSIZE1][READ_TIMES_ROW * LSIZE0 + 1];
#ifdef BORDER_CONSTANT
int end_addr = mad24(src_whole_rows - 1, src_step_in_pixel, src_whole_cols);
for (int i = 0; i < READ_TIMES_ROW; ++i)
{
int current_addr = mad24(i, LSIZE0 << 2, start_addr);
current_addr = current_addr < end_addr && current_addr > 0 ? current_addr : 0;
temp[i] = *(__global const uchar4 *)&src[current_addr];
}
#ifdef BORDER_ISOLATED
for (int i = 0; i < READ_TIMES_ROW; ++i)
{
temp[i].x = ELEM(start_x+i*LSIZE0*4,   src_offset_x, src_offset_x + src_cols, 0,         temp[i].x);
temp[i].y = ELEM(start_x+i*LSIZE0*4+1, src_offset_x, src_offset_x + src_cols, 0,         temp[i].y);
temp[i].z = ELEM(start_x+i*LSIZE0*4+2, src_offset_x, src_offset_x + src_cols, 0,         temp[i].z);
temp[i].w = ELEM(start_x+i*LSIZE0*4+3, src_offset_x, src_offset_x + src_cols, 0,         temp[i].w);
temp[i]   = ELEM(start_y,              src_offset_y, src_offset_y + src_rows, (uchar4)0, temp[i]);
}
#else
for (int i = 0; i < READ_TIMES_ROW; ++i)
{
temp[i].x = ELEM(start_x+i*LSIZE0*4,   0, src_whole_cols, 0,         temp[i].x);
temp[i].y = ELEM(start_x+i*LSIZE0*4+1, 0, src_whole_cols, 0,         temp[i].y);
temp[i].z = ELEM(start_x+i*LSIZE0*4+2, 0, src_whole_cols, 0,         temp[i].z);
temp[i].w = ELEM(start_x+i*LSIZE0*4+3, 0, src_whole_cols, 0,         temp[i].w);
temp[i]   = ELEM(start_y,              0, src_whole_rows, (uchar4)0, temp[i]);
}
#endif
#else
#ifdef BORDER_ISOLATED
int not_all_in_range = (start_x<src_offset_x) | (start_x + READ_TIMES_ROW*LSIZE0*4+4>src_offset_x + src_cols)| (start_y<src_offset_y) | (start_y >= src_offset_y + src_rows);
#else
int not_all_in_range = (start_x<0) | (start_x + READ_TIMES_ROW*LSIZE0*4+4>src_whole_cols)| (start_y<0) | (start_y >= src_whole_rows);
#endif
int4 index[READ_TIMES_ROW], addr;
int s_y;
if (not_all_in_range)
{
for (int i = 0; i < READ_TIMES_ROW; ++i)
{
index[i] = (int4)(mad24(i, LSIZE0 << 2, start_x)) + (int4)(0, 1, 2, 3);
#ifdef BORDER_ISOLATED
EXTRAPOLATE(index[i].x, src_offset_x, src_offset_x + src_cols);
EXTRAPOLATE(index[i].y, src_offset_x, src_offset_x + src_cols);
EXTRAPOLATE(index[i].z, src_offset_x, src_offset_x + src_cols);
EXTRAPOLATE(index[i].w, src_offset_x, src_offset_x + src_cols);
#else
EXTRAPOLATE(index[i].x, 0, src_whole_cols);
EXTRAPOLATE(index[i].y, 0, src_whole_cols);
EXTRAPOLATE(index[i].z, 0, src_whole_cols);
EXTRAPOLATE(index[i].w, 0, src_whole_cols);
#endif
}
s_y = start_y;
#ifdef BORDER_ISOLATED
EXTRAPOLATE(s_y, src_offset_y, src_offset_y + src_rows);
#else
EXTRAPOLATE(s_y, 0, src_whole_rows);
#endif
for (int i = 0; i < READ_TIMES_ROW; ++i)
{
addr = mad24((int4)s_y, (int4)src_step_in_pixel, index[i]);
temp[i].x = src[addr.x];
temp[i].y = src[addr.y];
temp[i].z = src[addr.z];
temp[i].w = src[addr.w];
}
}
else
{
for (int i = 0; i < READ_TIMES_ROW; ++i)
temp[i] = *(__global uchar4*)&src[mad24(i, LSIZE0 << 2, start_addr)];
}
#endif
for (int i = 0; i < READ_TIMES_ROW; ++i)
LDS_DAT[l_y][mad24(i, LSIZE0, l_x)] = temp[i];
barrier(CLK_LOCAL_MEM_FENCE);
sum = convertDstVec(vload4(0,(__local uchar *)&LDS_DAT[l_y][l_x]+RADIUSX+offset)) * mat_kernelrow[RADIUSX];
for (int i = 1; i <= RADIUSX; ++i)
{
temp[0] = vload4(0, (__local uchar*)&LDS_DAT[l_y][l_x] + RADIUSX + offset - i);
temp[1] = vload4(0, (__local uchar*)&LDS_DAT[l_y][l_x] + RADIUSX + offset + i);
#if (defined(INTEGER_ARITHMETIC) && !INTEL_DEVICE)
sum += mad24(convertDstVec(temp[0]), mat_kernelrow[RADIUSX-i], convertDstVec(temp[1]) * mat_kernelrow[RADIUSX + i]);
#else
sum += mad(convertDstVec(temp[0]), mat_kernelrow[RADIUSX-i], convertDstVec(temp[1]) * mat_kernelrow[RADIUSX + i]);
#endif
}
start_addr = mad24(y, dst_step_in_pixel, x);
if ((x+3<dst_cols) & (y<dst_rows))
*(__global dstT4*)&dst[start_addr] = sum;
else if ((x+2<dst_cols) && (y<dst_rows))
{
dst[start_addr] = sum.x;
dst[start_addr+1] = sum.y;
dst[start_addr+2] = sum.z;
}
else if ((x+1<dst_cols) && (y<dst_rows))
{
dst[start_addr] = sum.x;
dst[start_addr+1] = sum.y;
}
else if (x<dst_cols && y<dst_rows)
dst[start_addr] = sum.x;
}
__kernel void row_filter(__global const uchar * src, int src_step, int src_offset_x, int src_offset_y,
int src_cols, int src_rows, int src_whole_cols, int src_whole_rows,
        __global uchar * dst, int dst_step, int dst_cols, int dst_rows,
int radiusy)
{
int x = get_global_id(0);
int y = get_global_id(1);
int l_x = get_local_id(0);
int l_y = get_local_id(1);
int start_x = x + src_offset_x - RADIUSX;
int start_y = y + src_offset_y - radiusy;
int start_addr = mad24(start_y, src_step, start_x * SRCSIZErow);
dstTrow sum;
srcTrow temp[READ_TIMES_ROW];
__local srcTrow LDS_DAT[LSIZE1][READ_TIMES_ROW * LSIZE0 + 1];
#ifdef BORDER_CONSTANT
int end_addr = mad24(src_whole_rows - 1, src_step, src_whole_cols * SRCSIZErow);
for (int i = 0; i < READ_TIMES_ROW; i++)
{
int current_addr = mad24(i, LSIZE0 * SRCSIZErow, start_addr);
current_addr = current_addr < end_addr && current_addr >= 0 ? current_addr : 0;
temp[i] = loadpixrow(src + current_addr);
}
#ifdef BORDER_ISOLATED
for (int i = 0; i < READ_TIMES_ROW; ++i)
{
temp[i] = ELEM(mad24(i, LSIZE0, start_x), src_offset_x, src_offset_x + src_cols, (srcTrow)(0), temp[i]);
temp[i] = ELEM(start_y,                   src_offset_y, src_offset_y + src_rows, (srcTrow)(0), temp[i]);
}
#else
for (int i = 0; i < READ_TIMES_ROW; ++i)
{
temp[i] = ELEM(mad24(i, LSIZE0, start_x), 0, src_whole_cols, (srcTrow)(0), temp[i]);
temp[i] = ELEM(start_y,                   0, src_whole_rows, (srcTrow)(0), temp[i]);
}
#endif
#else
int index[READ_TIMES_ROW], s_x, s_y;
for (int i = 0; i < READ_TIMES_ROW; ++i)
{
s_x = mad24(i, LSIZE0, start_x);
s_y = start_y;
#ifdef BORDER_ISOLATED
EXTRAPOLATE(s_x, src_offset_x, src_offset_x + src_cols);
EXTRAPOLATE(s_y, src_offset_y, src_offset_y + src_rows);
#else
EXTRAPOLATE(s_x, 0, src_whole_cols);
EXTRAPOLATE(s_y, 0, src_whole_rows);
#endif
index[i] = mad24(s_y, src_step, s_x * SRCSIZErow);
}
for (int i = 0; i < READ_TIMES_ROW; ++i)
temp[i] = loadpixrow(src + index[i]);
#endif
for (int i = 0; i < READ_TIMES_ROW; ++i)
LDS_DAT[l_y][mad24(i, LSIZE0, l_x)] = temp[i];
barrier(CLK_LOCAL_MEM_FENCE);
sum = convertToDstTrow(LDS_DAT[l_y][l_x + RADIUSX]) * mat_kernelrow[RADIUSX];
for (int i = 1; i <= RADIUSX; ++i)
{
temp[0] = LDS_DAT[l_y][l_x + RADIUSX - i];
temp[1] = LDS_DAT[l_y][l_x + RADIUSX + i];
#if (defined(INTEGER_ARITHMETIC) && !INTEL_DEVICE)
sum += mad24(convertToDstTrow(temp[0]), mat_kernelrow[RADIUSX - i], convertToDstTrow(temp[1]) * mat_kernelrow[RADIUSX + i]);
#else
sum += mad(convertToDstTrow(temp[0]), mat_kernelrow[RADIUSX - i], convertToDstTrow(temp[1]) * mat_kernelrow[RADIUSX + i]);
#endif
}
if (x < dst_cols && y < dst_rows)
{
start_addr = mad24(y, dst_step, x * DSTSIZErow);
storepixrow(sum, dst + start_addr);
}
}



// ~/opencv-3.2.0/builds/modules/imgproc/opencl_kernels_imgproc.cpp
//OPENCL SEP COL FILTER 2D KERNEL:

__constant srcT1col mat_kernelcol[] = { COEFF };
__kernel void col_filter(__global const uchar * src, int src_step, int src_offset, int src_whole_rows, int src_whole_cols,
        __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols, float delta)
{
int x = get_global_id(0);
int y = get_global_id(1);
int l_x = get_local_id(0);
int l_y = get_local_id(1);
int start_addr = mad24(y, src_step, x * SRCSIZEcol);
int end_addr = mad24(src_whole_rows - 1, src_step, src_whole_cols * SRCSIZEcol);
srcTcol sum, temp[READ_TIMES_COL];
__local srcTcol LDS_DAT[LSIZE1 * READ_TIMES_COL][LSIZE0 + 1];
for (int i = 0; i < READ_TIMES_COL; ++i)
{
int current_addr = mad24(i, LSIZE1 * src_step, start_addr);
current_addr = current_addr < end_addr ? current_addr : 0;
temp[i] = loadpixcol(src + current_addr);
}
for (int i = 0; i < READ_TIMES_COL; ++i)
LDS_DAT[mad24(i, LSIZE1, l_y)][l_x] = temp[i];
barrier(CLK_LOCAL_MEM_FENCE);
sum = LDS_DAT[l_y + RADIUSY][l_x] * mat_kernelcol[RADIUSY];
for (int i = 1; i <= RADIUSY; ++i)
{
temp[0] = LDS_DAT[l_y + RADIUSY - i][l_x];
temp[1] = LDS_DAT[l_y + RADIUSY + i][l_x];
#if (defined(INTEGER_ARITHMETIC) && !INTEL_DEVICE)
sum += mad24(temp[0],mat_kernelcol[RADIUSY - i], temp[1] * mat_kernelcol[RADIUSY + i]);
#else
sum += mad(temp[0], mat_kernelcol[RADIUSY - i], temp[1] * mat_kernelcol[RADIUSY + i]);
#endif
}
#ifdef INTEGER_ARITHMETIC
#ifdef INTEL_DEVICE
sum = (sum + (1 << (SHIFT_BITS-1))) / (1 << SHIFT_BITS);
#else
sum = (sum + (1 << (SHIFT_BITS-1))) >> SHIFT_BITS;
#endif
#endif
if (x < dst_cols && y < dst_rows)
{
start_addr = mad24(y, dst_step, mad24(DSTSIZEcol, x, dst_offset));
storepixcol(convertToDstTcol(sum + (srcTcol)(delta)), dst + start_addr);
}
}
