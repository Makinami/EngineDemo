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

#include "CDLODQuadTree.h"

#include "PostFX.h"

using namespace std;
using namespace DirectX;

class OceanClass
{
public:
	OceanClass();
	~OceanClass();

	HRESULT Init(ID3D11Device1* &device, ID3D11DeviceContext1* &mImmediateContext);

	void Update(ID3D11DeviceContext1* &mImmediateContext, float dt, DirectionalLight& light, std::shared_ptr<CameraClass> Camera);

	void Draw(ID3D11DeviceContext1* &mImmediateContext, std::shared_ptr<CameraClass> Camera, DirectionalLight& light);

	void DrawPost(ID3D11DeviceContext1* &mImmediateContext, std::unique_ptr<PostFX::Canvas> const& Canvas, std::shared_ptr<CameraClass> Camera, DirectionalLight& light);

	void ChangedWinSize(ID3D11Device1* &device, int width, int height);

	void Release();

private:
	HRESULT CompileShadersAndInputLayout(ID3D11Device1* &device);

	HRESULT CreateConstantBuffers(ID3D11Device1* &device);

	HRESULT CreateDataResources(ID3D11Device1* &device);

	HRESULT CreateScreenMesh(ID3D11Device1* &device);

	HRESULT CreateDiscMesh(ID3D11Device1* &device);

	HRESULT CreateGridMesh(ID3D11Device1* &device);

	HRESULT CreateSamplerRasterDepthStencilStates(ID3D11Device1* &device);

private:
	void Simulate(ID3D11DeviceContext1* &mImmediateContext);

	void BuildInstanceBuffer(ID3D11DeviceContext1* &mImmediateContext, std::shared_ptr<CameraClass> Camera);

	void DivideTile(XMFLOAT2 pos, float edge, int num);

private:
	Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> spectrumSRV;

	Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> varianceSRV;
	Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> varianceUAV;

	vector< Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> > wavesSRV;
	vector< Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> > wavesUAV;

	vector< Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> > turbulenceSRV;
	vector< Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> > turbulenceUAV;

	Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> noiseSRV;

	Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> fresnelSRV;

	Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> foamSRV;

	Microsoft::WRL::ComPtr<ID3D11ComputeShader> slopeVarianceCS;
	Microsoft::WRL::ComPtr<ID3D11ComputeShader> initFFTCS;
	Microsoft::WRL::ComPtr<ID3D11ComputeShader> fftCS;
	Microsoft::WRL::ComPtr<ID3D11ComputeShader> JacobianInjectCS;
	Microsoft::WRL::ComPtr<ID3D11ComputeShader> dissipateTurbulanceCS;

	vector< Microsoft::WRL::ComPtr<ID3D11Buffer> > constCB;

	Microsoft::WRL::ComPtr<ID3D11Buffer> perFrameCB;

	struct TileData
	{
		XMFLOAT2 offset;
		int lod;
		float pad;
	};

	std::vector<TileData> instanceData;
	Microsoft::WRL::ComPtr<ID3D11Buffer> gridMeshVB;
	Microsoft::WRL::ComPtr<ID3D11Buffer> gridMeshIB;
	Microsoft::WRL::ComPtr<ID3D11Buffer> gridInstancesVB;

	Microsoft::WRL::ComPtr<ID3D11Buffer> screenMeshVB;
	Microsoft::WRL::ComPtr<ID3D11Buffer> screenMeshIB;

	Microsoft::WRL::ComPtr<ID3D11Buffer> discMeshVB;
	Microsoft::WRL::ComPtr<ID3D11Buffer> discMeshIB;

	UINT stride;
	UINT offset;

	Microsoft::WRL::ComPtr<ID3D11InputLayout> mInputLayout;
	Microsoft::WRL::ComPtr<ID3D11VertexShader> mVertexShader;
	Microsoft::WRL::ComPtr<ID3D11HullShader> mHullShader;
	Microsoft::WRL::ComPtr<ID3D11DomainShader> mDomainShader;
	Microsoft::WRL::ComPtr<ID3D11PixelShader> mPixelShader;

	Microsoft::WRL::ComPtr<ID3D11VertexShader> mVS2;
	ID3D11HullShader* mHS2;
	ID3D11DomainShader* mDS2;
	ID3D11PixelShader* mPS2;
	ID3D11PixelShader* mPS3;

	Microsoft::WRL::ComPtr<ID3D11DomainShader> mGerstnerDS;

	Microsoft::WRL::ComPtr<ID3D11VertexShader> mQuadVS;
	Microsoft::WRL::ComPtr<ID3D11InputLayout> mQuadIL;

	Microsoft::WRL::ComPtr<ID3D11SamplerState> mSamplerAnisotropic;
	Microsoft::WRL::ComPtr<ID3D11SamplerState> mSamplerClamp;
	Microsoft::WRL::ComPtr<ID3D11SamplerState> mSamplerBilinear;

	Microsoft::WRL::ComPtr<ID3D11RasterizerState> mRastStateFrame;
	Microsoft::WRL::ComPtr<ID3D11RasterizerState> mRastStateSolid;
	Microsoft::WRL::ComPtr<ID3D11DepthStencilState> mDepthStencilState;

	CDLODFlatQuadTree oceanQuadTree;

	int indicesPerRow;
	int screenGridSize;
	int indicesToRender;
	char turbulenceTextReadId;
	char turbulenceTextWriteId;

	struct {
		XMFLOAT4X4 screenToCamMatrix;
		XMFLOAT4X4 camToWorldMatrix;
		XMFLOAT4X4 worldToScreenMatrix;
		XMFLOAT3 camPos;
		float time;
		XMFLOAT3 sunDir;
		float dt;
		float coneAngle;
		XMFLOAT2 gridSize;
		float lambdaJ;
		XMFLOAT2 sigma2;
		float lambdaV;
		float scale;
		XMFLOAT3 camLookAt;
		float pad;
		XMFLOAT4 gProj;
	} perFrameParams;

	struct LODLevel
	{

	};

private:
	void getSpectrumSample(int i, int j, float lengthScale, float kMin, float* result);
	float getSlopeVariance(float kx, float ky, float * spectrum);

	float spectrum(float kx, float ky, bool omnispectrum = false);
	float omega(float k);
	float inline sqr(float x) { return x * x; };

	float Fresnel(float dot, float n1, float n2, bool schlick = true);

private:
	UINT FFT_SIZE;
	float GRID_SIZE[4];

	UINT16 varianceRes;

	// spectrum
	float windSpeed;
	float waveAge;
	float cm;
	float km;
	float spectrumGain;

	float time;

	mt19937 mt;
	uniform_real_distribution<float> rand01;

	int screenWidth;
	int screenHeight;

	////

	struct initFFTCBType
	{
		XMFLOAT4 INVERSE_GRID_SIZE;
		float time;
		XMFLOAT3 pad;
	};
	initFFTCBType initFFTParams;

	ID3D11Buffer* initFFTCB;

	int screen;

	vector<XMFLOAT4X4> instances[3];
	XMFLOAT3 cameraPos;

};

