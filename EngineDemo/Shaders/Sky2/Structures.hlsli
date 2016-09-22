#ifdef __cplusplus
#pragma once
#endif

#ifdef __cplusplus

#define float2 XMFLOAT2;
#define float3 XMFLOAT3;
#define float4 XMFLOAT4;
#define uint UINT;

#endif

#ifdef __cplusplus
#   define CHECK_STRUCT_ALIGNMENT(s) static_assert( sizeof(s) % 16 == 0, "sizeof("#s") is not multiple of 16" );
#else
#   define CHECK_STRUCT_ALIGNMENT(s)
#endif

static const float groundR = 6360.0;
static const float topR = 6420.0;

static const int TRANSMITTANCE_W = 256;
static const int TRANSMITTANCE_H = 64;

static const int SKY_W = 64;
static const int SKY_H = 16;

static const unsigned int RES_ALT = 32;
static const unsigned int RES_VZ = 128;
static const unsigned int RES_SZ = 32;
static const unsigned int RES_VS = 8;