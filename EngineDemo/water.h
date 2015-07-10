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
#include "FFT.h"

using namespace std;
using namespace DirectX;

struct vertex_ocean
{
	float x, y, z; // vertex
	float nx, ny, nz; // normal
	float a, b, c; //htilde0
	float _a, _b, _c; // htilde0mk conjugate
	float ox, oy, oz; // original position
};

struct complex_vector_normal
{
	complex<float> h; // wave height
	XMFLOAT2 D; // displacement
	XMFLOAT3 n; // normal
};

class WaterClass : public HasLogger
{
public:
	struct Vertex
	{
		XMFLOAT3 Pos;
		XMFLOAT3 Normal;
		XMFLOAT3 hTilde0;
		XMFLOAT3 hTilde0mkconj;
		XMFLOAT3 Original;
	};

public:
	WaterClass();
	~WaterClass();

	bool Init(ID3D11Device1* device, ID3D11DeviceContext1* dc);

	void Draw(ID3D11DeviceContext1* mImmediateContext, std::shared_ptr<CameraClass> Camera);
	void DrawWithShadow(ID3D11DeviceContext1* mImmediateContext, std::shared_ptr<CameraClass> Camera, ID3D11ShaderResourceView* shadowmap);

	float dispersion(int n_prime, int m_prime); // deep water
	float phillips(int n_prime, int m_prime); // phillips spectrum
	complex<float> hTilde_0(int n_prime, int m_prime);
	complex<float> hTilde(float t, int n_prime, int m_prime);
	complex_vector_normal h_D_and_n(XMFLOAT2 x, float t);
	void evaluateWaves(float t);
	void evaluateWavesFFT(float t);
	void evaluateWavesGPU(float t);

private:
	struct MatrixBufferType
	{
		XMMATRIX gWorld;
	};

private:
	void BuildQuadPatchVB(ID3D11Device1* device);
	bool BuildQuadPatchIB(ID3D11Device1* device);
	bool CreateInputLayoutAndShaders(ID3D11Device1* device);

	ID3D11Buffer* mQuadPatchVB;
	ID3D11Buffer* mQuadPatchIB;

	ID3D11Buffer* MatrixBuffer;

	ID3D11InputLayout* mInputLayout;
	ID3D11VertexShader* mVertexShader;
	ID3D11PixelShader* mPixelShader;

	XMFLOAT4X4 mWorld;

	float g; // gravity constant
	int N, Nplus1; // dimension -- N should be a power of 2
	float A; // phillips spectrum parameter -- affect heights of waves
	XMFLOAT2 w; // wind parameter
	float length; // length parameter
	complex<float> *h_tilde, *h_tilde_slopex, *h_tilde_slopez, *h_tilde_dx, *h_tilde_dz; // for fft
	cFFT *fft;

	default_random_engine generator;
	normal_distribution<float> distribution;

	float time;

	ID3D11RasterizerState* mRastStateFrame;

	vector<vertex_ocean> vertices;
	vector<UINT>indices;
	UINT indices_count;

	ID3DX11FFT* mFFTDevice;
};