#pragma once

#include <d3d11_1.h>

#define ReleaseCOM(x) { if(x){ x->Release(); x = nullptr; } }
#define SafeDelete(x) { delete x; x = nullptr; }

#pragma warning(disable:4834)
#include <DirectXMath.h>
#pragma warning(default:4838)

#include <random>
#include <vector>
#include <memory>

#include <wrl\client.h>

#include "cameraclass.h"
#include "Lights.h"

using namespace std;
using namespace DirectX;

class OceanClass
{
public:
	OceanClass();
	~OceanClass();

	HRESULT Init(ID3D11Device1* &device, ID3D11DeviceContext1* &mImmediateContext);

	void Update(ID3D11DeviceContext1* &mImmediateContext, float dt, std::shared_ptr<CameraClass> Camera);

	void Draw(ID3D11DeviceContext1* &mImmediateContext, std::shared_ptr<CameraClass> Camera, DirectionalLight& light );

	void ChangedWinSize(ID3D11Device1* &device, int width, int height);

	void Release();

private:
	HRESULT CompileShadersAndInputLayout(ID3D11Device1* &device);

	HRESULT CreateConstantBuffers(ID3D11Device1* &device);

	HRESULT CreateDataResources(ID3D11Device1* &device);

	HRESULT CreateScreenMesh(ID3D11Device1* &device);

	HRESULT CreateSamplerRasterDepthStencilStates(ID3D11Device1* &device);

private:
	void Simulate(ID3D11DeviceContext1* &mImmediateContext);

private:
	Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> spectrumSRV;

	vector< Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> > wavesSRV;
	vector< Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> > wavesUAV;

	Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> turbulenceSRV;
	Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> turbulenceUAV;

	Microsoft::WRL::ComPtr<ID3D11ComputeShader> initFFTCS;
	Microsoft::WRL::ComPtr<ID3D11ComputeShader> fftCS;
	Microsoft::WRL::ComPtr<ID3D11ComputeShader> injectTurbulanceCS;
	Microsoft::WRL::ComPtr<ID3D11ComputeShader> dissipateTurbulanceCS;

	vector< Microsoft::WRL::ComPtr<ID3D11Buffer> > constCB;

	Microsoft::WRL::ComPtr<ID3D11Buffer> perFrameCB;

	Microsoft::WRL::ComPtr<ID3D11Buffer> screenMeshVB;
	Microsoft::WRL::ComPtr<ID3D11Buffer> screenMeshIB;
	UINT stride;
	UINT offset;

	Microsoft::WRL::ComPtr<ID3D11InputLayout> mInputLayout;
	Microsoft::WRL::ComPtr<ID3D11VertexShader> mVertexShader;
	Microsoft::WRL::ComPtr<ID3D11PixelShader> mPixelShader;

	Microsoft::WRL::ComPtr<ID3D11SamplerState> mSamplerAnisotropic;
	Microsoft::WRL::ComPtr<ID3D11RasterizerState> mRastStateFrame;
	Microsoft::WRL::ComPtr<ID3D11RasterizerState> mRastStateSolid;
	Microsoft::WRL::ComPtr<ID3D11DepthStencilState> mDepthStencilState;

	int indicesPerRow;
	int screenGridSize;
	int indicesToRender;

	struct {
		XMFLOAT4X4 screenToCamMatrix;
		XMFLOAT4X4 camToWorldMatrix;
		XMFLOAT4X4 worldToScreenMatrix;
		XMFLOAT3 camPos;
		float time;
		float dt;
		float screendy;
		XMFLOAT2 gridSize;
	} perFrameParams;

private:
	void getSpectrumSample(int i, int j, float lengthScale, float kMin, float* result);

	float spectrum(float kx, float ky, bool omnispectrum = false);
	float omega(float k);
	float inline sqr(float x) { return x * x; };

private:
	UINT FFT_SIZE;
	float GRID_SIZE[4];

	// spectrum
	float windSpeed;
	float waveAge;
	float cm;
	float km;
	float amplitude;

	float time;

	mt19937 mt;
	uniform_real_distribution<float> rand01;

	int screenWidth;
	int screenHeight;

};

