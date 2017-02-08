#pragma once

#if defined(DEBUG) || defined(_DEBUG)
#define CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif

#include <d3d11_1.h>

#pragma warning(disable:4838)
#include <DirectXMath.h>
#pragma warning(default:4838)

#include <DirectXPackedVector.h>

#include <d3dcsx.h>
#pragma comment(lib,  "D3dcsx.lib")

#define ReleaseCOM(x) { if(x){ x->Release(); x = nullptr; } }
#define SafeDelete(x) { delete x; x = nullptr; }

#include <vector>
#include <complex>
#include <random>
#include <string>

#include "loggerclass.h"
#include "cameraclass.h"
#include "Lights.h"

using namespace std;
using namespace DirectX;

class WaterClass : public HasLogger
{
public:
	struct Vertex
	{
		XMFLOAT3 Pos;
		XMFLOAT2 Tex;
	};

public:
	WaterClass();
	~WaterClass();

	bool Init(ID3D11Device1* device, ID3D11DeviceContext1* dc);

	void Draw(ID3D11DeviceContext1 * mImmediateContext, std::shared_ptr<CameraClass> Camera, DirectionalLight& light, ID3D11ShaderResourceView * ShadowMap);

	void evaluateWavesGPU(float t, ID3D11DeviceContext1 * mImmediateContext);

private:
	struct cbPerFrameVSType
	{
		XMMATRIX gWorld;
	};

	struct cbPerFrameDSType
	{
		XMMATRIX gViewProj;
		XMMATRIX gShadowTrans;
	};

	struct cbPerFrameHSType
	{
		XMFLOAT4 gWorldFrustumPlanes[6];

		XMFLOAT3 gEyePosW;

		// When distance is minimum, the tessellation is maximum.
		// When distance is maximum, the tessellation is minimum.
		float gMinDist;
		float gMaxDist;

		// Exponents for power of 2 tessellation.  The tessellation
		// range is [2^(gMinTess), 2^(gMaxTess)].  Since the maximum
		// tessellation is 64, this means gMaxTess can be at most 6
		// since 2^6 = 64.
		float gMinTess;
		float gMaxTess;

		int gFrustumCull;
	};

	struct cbPerFramePSType
	{
		XMMATRIX gViewProj;

		//DirectionalLightStruct gDirLight;
		XMFLOAT3 gEyePosW;

		float gTexelCellSpaceU;
		float gTexelCellSpaceV;
		float gWorldCellSpace;

		float padding[2];
	};

	struct FFTInitialType
	{
		XMFLOAT2 h_0;
		XMFLOAT2 h_0conj;
		float dispersion;
	};

	struct TimeBufferCS
	{
		XMFLOAT4 time;
	};

	struct FFTParameters
	{
		UINT col_row_pass[4];
	};
private:
	void BuildQuadPatchVB(ID3D11Device1* device);
	bool BuildQuadPatchIB(ID3D11Device1* device);
	bool CreateInputLayoutAndShaders(ID3D11Device1* device);

	bool CreateInitialDataResource(ID3D11Device1* device);

	ID3D11Buffer* mQuadPatchVB;
	ID3D11Buffer* mQuadPatchIB;

	ID3D11Buffer* cbPerFrameHS;
	ID3D11Buffer* cbPerFrameDS;
	ID3D11Buffer* cbPerFramePS;
	ID3D11Buffer* cbPerFrameVS;
	ID3D11Buffer* FFTPrepBuffer;
	ID3D11Buffer* FFTBuffer;

	ID3D11Buffer* mFFTInitial;
	ID3D11ShaderResourceView* mFFTInitialSRV;
	ID3D11UnorderedAccessView* mFFTUAV[2][2];
	ID3D11ShaderResourceView* mFFTSRV[2][2];

	ID3D11InputLayout* mInputLayout;
	ID3D11VertexShader* mVertexShader;
	ID3D11HullShader* mHullShader;
	ID3D11DomainShader* mDomainShader;
	ID3D11PixelShader* mPixelShader;
	vector<ID3D11ComputeShader*> mComputeShader;

	UINT mNumPatchVertices;
	UINT mNumPatchQuadFaces;

	UINT mNumPatchVertRows;
	UINT mNumPatchVertCols;

	XMFLOAT4X4 mWorld;

	const float g;
	int N, Nplus1; // dimension -- N should be a power of 2
	float A; // phillips spectrum parameter -- affect heights of waves
	XMFLOAT2 w; // wind parameter
	float length; // length parameter

	default_random_engine generator;
	normal_distribution<float> distribution;

	float time;

	ID3D11RasterizerState* mRastStateFrame;
	ID3D11SamplerState** mSamplerStates;

	vector<Vertex> vertices;
	vector<UINT> indices;
	UINT indices_count;

	// helpers
	float PhillipsSpectrum(int n, int m);
	complex<float> hTilde_0(int n, int m);
	float Dispersion(int n, int m);

	// performance ids
	char computeFFTPrf;
	char drawPrf;
};

// TEMP
void ExtractFrustrumPlanes(XMFLOAT4 planes[6], CXMMATRIX M);