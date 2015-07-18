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

using namespace std;
using namespace DirectX;

class WaterClass20 : public HasLogger
{
public:
	struct Vertex
	{
		XMFLOAT3 Pos;
	};

public:
	WaterClass20();
	~WaterClass20();

	bool Init(ID3D11Device1* device, ID3D11DeviceContext1* dc);

	void Draw(ID3D11DeviceContext1* mImmediateContext, std::shared_ptr<CameraClass> Camera);

	void evaluateWavesGPU(float t, ID3D11DeviceContext1 * mImmediateContext);

private:
	struct MatrixBufferType
	{
		XMMATRIX gWorld;
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

	ID3D11Buffer* MatrixBuffer;
	ID3D11Buffer* FFTPrepBuffer;
	ID3D11Buffer* FFTBuffer;

	ID3D11Buffer* mFFTInitial;
	ID3D11ShaderResourceView* mFFTInitialSRV;
	ID3D11Buffer* mFFTOutput;
	ID3D11UnorderedAccessView* mFFTUAV[2][2];
	ID3D11ShaderResourceView* mFFTSRV[2][2];

	ID3D11InputLayout* mInputLayout;
	ID3D11VertexShader* mVertexShader;
	ID3D11PixelShader* mPixelShader;
	vector<ID3D11ComputeShader*> mComputeShader;

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

	vector<Vertex> vertices;
	vector<UINT> indices;
	UINT indices_count;

	// helpers
	float PhillipsSpectrum(int n, int m);
	complex<float> hTilde_0(int n, int m);
	float Dispersion(int n, int m);
};