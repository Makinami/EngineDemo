#pragma once

#include <d3d11_1.h>

#define ReleaseCOM(x) { if(x){ x->Release(); x = nullptr; } }
#define SafeDelete(x) { delete x; x = nullptr; }

#pragma warning(disable:4838)
#include <DirectXMath.h>
#pragma warning(default:4838)

#include <wrl\client.h>

#include <random>

#include <memory>

#include "Performance.h"
#include "cameraclass.h"
#include "Lights.h"

using namespace std;
using namespace DirectX;
using namespace Microsoft::WRL;

struct D
{
	void operator()(IUnknown *p)
	{
		if (p) {
			p->Release();
			p = nullptr;
		}
	}
};

class WaterBruneton : public Debug::HasPerformance
{
public:
	WaterBruneton();
	~WaterBruneton();

	bool Init(ID3D11Device1* device, ID3D11DeviceContext1* mImmediateContext);

	void Draw(ID3D11DeviceContext1* mImmediateContext, std::shared_ptr<CameraClass> Camera, DirectionalLight& light);

	void EvaluateWaves(float t, ID3D11DeviceContext1* mImmediateContext);

	void BEvelWater(float t, ID3D11DeviceContext1* mImmediateContext);

	bool BCreatespectrumEtc(ID3D11Device1* device);

	// temp 
	ID3D11ShaderResourceView* getFFTWaves() { return fftWavesSRV; };

	ID3D11Device1* mDevice;

private:
	// Initialization functions
	bool CreateDataResources(ID3D11Device1* device);
	bool CreateInputLayoutAndShaders(ID3D11Device1* device);

	void ComputeVarianceText(ID3D11DeviceContext1* mImmediateContext);

	void GenerateScreenMesh(/*ID3D11Device1* device*/);

	// Helpers 
	void getSpectrumSample(int i, int j, float lengthScale, float kMin, float* result);
	float spectrum(float kx, float ky, bool omnispectrum = false);
	float omega(float k);
	float inline sqr(float x) { return x * x; };
	float getSlopeVariance(float kx, float ky, float* spectrumSample);

private:
	ID3D11ShaderResourceView* spectrumSRV;
	//ComPtr<ID3D11UnorderedAccessView> spectrumUAV;

	ID3D11ShaderResourceView* slopeVarianceSRV;
	ID3D11UnorderedAccessView* slopeVarianceUAV;

	ID3D11ShaderResourceView* fftWavesSRV;
	ID3D11UnorderedAccessView* fftWavesUAV;
	ID3D11ShaderResourceView* fftWavesSRVTemp;
	ID3D11UnorderedAccessView* fftWavesUAVTemp;

	ID3D11ComputeShader* variancesCS;
	ID3D11ComputeShader* initFFTCS;
	ID3D11ComputeShader* fftCS;

	ID3D11ComputeShader* BinitFFTCS;
	ID3D11ComputeShader* Bfftx;
	ID3D11ComputeShader* Bffty;

	ID3D11ShaderResourceView* spectrum12SRV;
	ID3D11ShaderResourceView* spectrum34SRV;

	ID3D11ShaderResourceView* BbutterflySRV;

	ID3D11InputLayout* mInputLayout;
	ID3D11VertexShader* mVertexShader;
	ID3D11PixelShader* mPixelShader;

	ID3D11Buffer* variancesCB;
	ID3D11Buffer* initFFTCB;
	ID3D11Buffer* fftCB;
	ID3D11Buffer* drawCB;

	ID3D11Buffer* mScreenMeshVB;
	ID3D11Buffer* mScreenMeshIB;

	ID3D11RasterizerState* mRastStateFrame;
	ID3D11SamplerState* mSamplerState;
	ID3D11SamplerState* mSamplerAnisotropic;
	ID3D11DepthStencilState* mDepthStencilStateSea;

	int mScreenWidth;
	int mScreenHeight;

	float screenGridSize;
	UINT mNumScreenMeshIndices;

	float horizon;

	const unsigned int FFT_SIZE;
	const float GRID_SIZE[4];
	const float N_SLOPE_VARIANCE;

	float WIND;
	float OMEGA;
	float A;
	bool propagate; // wave propagation?

	const float cm;
	const float km;

	float time;

	float theoreticalSlopeVariance;
	float totalSlopeVariance;

	random_device rd;
	mt19937 mt;
	uniform_real_distribution<float> rand01;

	struct variancesCBType
	{
		XMFLOAT4 GRID_SIZE;
		float slopeVarianceDelta;
		XMFLOAT3 pad;
	};
	variancesCBType variancesParams;

	struct initFFTCBType
	{
		XMFLOAT4 INVERSE_GRID_SIZE;
		float time;
		XMFLOAT3 pad;
	};
	initFFTCBType initFFTParams;

	struct fftCBType
	{
		enum { ROW_PASS, COL_PASS };
		unsigned int col_row_pass;
		XMFLOAT3 pad;
	} fftParams;
	//fftCBType fftParams;

	struct vertexShaderCBType
	{
		XMFLOAT4X4 screenToCamera;
		XMFLOAT4X4 cameraToWorld;
		XMFLOAT4X4 worldToScreen;
		XMFLOAT3 worldCamera;
		float normals;
		XMFLOAT3 worldSunDir;
		float choppy;
		XMFLOAT4 GRID_SIZE;
		XMFLOAT3 seaColour;
		float pad;
		XMFLOAT2 gridSize;
		XMFLOAT2 pad2;
	} drawParams;

	// performance ids
	char computeFFTPrf;
	char drawPrf;
};

