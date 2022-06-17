#include "map.h"

#include "Utilities\RenderViewTargetStack.h"
#include "Utilities\CreateShader.h"
#include "Utilities\MapResources.h"

#include "RenderStates.h"

MapClass::MapClass() :
	Terrain(nullptr),
	Water(nullptr),
	Sky(nullptr)
{
}

MapClass::~MapClass()
{
	Shutdown();
}

bool MapClass::Init(ID3D11Device1* device, ID3D11DeviceContext1 * dc)
{
	Sky = std::make_shared<SkyClass>();
	Sky->Init(device, dc);
	
	Clouds2 = std::make_shared<CloudsClass2>();
	Clouds2->Init(device, dc);

	Terrain = std::make_shared<TerrainClass>();
	TerrainClass::InitInfo tii;
	tii.HeightMapFilename = L"Textures/terrain.raw";
	tii.LayerMapFilename0 = L"Textures/grass.dds";
	tii.LayerMapFilename1 = L"Textures/darkdirt.dds";
	tii.LayerMapFilename2 = L"Textures/stone.dds";
	tii.LayerMapFilename3 = L"Textures/lightdirt.dds";
	tii.LayerMapFilename4 = L"Textures/snow.dds";
	tii.BlendMapFilename = L"Textures/blend.dds";
	tii.HeightScale = 100.0f;
	tii.HeightmapWidth = 2049;
	tii.HeightmapHeight = 2049;
	tii.CellSpacing = 0.5f;
	if (!Terrain->Init(device, dc, tii))
	{
		LogError(L"Failed to initiate terrain");
		return false;
	}

	Terrain2 = std::make_unique<TerrainClass2>();
	Terrain2->Init(device, dc);

	light.Ambient(XMFLOAT4(0.2f, 0.2f, 0.2f, 1.0f));
	light.Diffuse(XMFLOAT4(0.5f, 0.5f, 0.5f, 1.0f));
	light.Specular(XMFLOAT4(0.5f, 0.5f, 0.5f, 1.0f));
	light.Direction(XMFLOAT3(-0.707f, -0.707f, 0.0f));

	ShadowMap = std::make_unique<ShadowMapClass>(device, 2048, 2048);

	// DS
	D3D11_BUFFER_DESC matrixBufferDesc;
	matrixBufferDesc.Usage = D3D11_USAGE_DYNAMIC;
	matrixBufferDesc.ByteWidth = sizeof(MatrixBufferType);
	matrixBufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	matrixBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	matrixBufferDesc.MiscFlags = 0;
	matrixBufferDesc.StructureByteStride = 0;

	if (FAILED(device->CreateBuffer(&matrixBufferDesc, NULL, &MatrixBuffer))) return false;

	// CUBE
	vector<XMFLOAT4> cubeVertices;
	cubeVertices.push_back(XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f));
	cubeVertices.push_back(XMFLOAT4(1.0f, -1.0f, 1.0f, 1.0f));
	cubeVertices.push_back(XMFLOAT4(1.0f, 1.0f, -1.0f, 1.0f));
	cubeVertices.push_back(XMFLOAT4(1.0f, -1.0f, -1.0f, 1.0f));

	cubeVertices.push_back(XMFLOAT4(-1.0f, 1.0f, 1.0f, 0.1f));
	cubeVertices.push_back(XMFLOAT4(-1.0f, -1.0f, 1.0f, 0.1f));
	cubeVertices.push_back(XMFLOAT4(-1.0f, 1.0f, -1.0f, 0.1f));
	cubeVertices.push_back(XMFLOAT4(-1.0f, -1.0f, -1.0f, 0.1f));

	cubeVertices.push_back(XMFLOAT4(1.0f, 1.0f, 1.0f, 0.25f));
	cubeVertices.push_back(XMFLOAT4(-1.0f, 1.0f, 1.0f, 0.25f));
	cubeVertices.push_back(XMFLOAT4(1.0f, 1.0f, -1.0f, 0.25f));
	cubeVertices.push_back(XMFLOAT4(-1.0f, 1.0f, -1.0f, 0.25f));

	cubeVertices.push_back(XMFLOAT4(1.0f, -1.0f, 1.0f, 0.4f));
	cubeVertices.push_back(XMFLOAT4(-1.0f, -1.0f, 1.0f, 0.4f));
	cubeVertices.push_back(XMFLOAT4(1.0f, -1.0f, -1.0f, 0.4f));
	cubeVertices.push_back(XMFLOAT4(-1.0f, -1.0f, -1.0f, 0.4f));

	cubeVertices.push_back(XMFLOAT4(1.0f, 1.0f, 1.0f, 0.55f));
	cubeVertices.push_back(XMFLOAT4(-1.0f, 1.0f, 1.0f, 0.55f));
	cubeVertices.push_back(XMFLOAT4(1.0f, -1.0f, 1.0f, 0.55f));
	cubeVertices.push_back(XMFLOAT4(-1.0f, -1.0f, 1.0f, 0.55f));

	cubeVertices.push_back(XMFLOAT4(1.0f, 1.0f, 1.0f, 0.7f));
	cubeVertices.push_back(XMFLOAT4(-1.0f, 1.0f, 1.0f, 0.7f));
	cubeVertices.push_back(XMFLOAT4(1.0f, -1.0f, 1.0f, 0.7f));
	cubeVertices.push_back(XMFLOAT4(-1.0f, -1.0f, 1.0f, 0.7f));

	D3D11_BUFFER_DESC vbd;
	vbd.Usage = D3D11_USAGE_IMMUTABLE;
	vbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	vbd.CPUAccessFlags = 0;
	vbd.MiscFlags = 0;
	vbd.StructureByteStride = 0;
	vbd.ByteWidth = sizeof(cubeVertices[0])*cubeVertices.size();

	D3D11_SUBRESOURCE_DATA vinitData;
	vinitData.pSysMem = &cubeVertices[0];
	device->CreateBuffer(&vbd, &vinitData, &mCubeVB);

	vector<USHORT> cubeIndices{ 0,1,2,1,2,3,4,5,6,5,6,7,8,9,10,9,10,11,12,13,14,13,14,15,16,17,18,17,18,19,20,21,22,21,22,23 };

	D3D11_BUFFER_DESC ibd;
	ibd.Usage = D3D11_USAGE_IMMUTABLE;
	ibd.BindFlags = D3D11_BIND_INDEX_BUFFER;
	ibd.CPUAccessFlags = 0;
	ibd.MiscFlags = 0;
	ibd.StructureByteStride = 0;
	ibd.ByteWidth = sizeof(cubeIndices[0])*cubeIndices.size();

	D3D11_SUBRESOURCE_DATA iinitData;
	iinitData.pSysMem = &cubeIndices[0];
	device->CreateBuffer(&ibd, &iinitData, &mCubeIB);

	CreatePSFromFile(L"..\\Debug\\BasicPS.cso", device, mCubePS);

	D3D11_INPUT_ELEMENT_DESC cubeVertexDesc[] = {
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 }
	};

	CreateVSAndInputLayout(L"..\\Debug\\cubeVS.cso", device, mCubeVS, cubeVertexDesc, 1, mCubeIL);

	return true;
}

void MapClass::Shutdown()
{
	ReleaseCOM(mScreenQuadIB);
	ReleaseCOM(mScreenQuadVB);

	ReleaseCOM(mDebugIL);
	ReleaseCOM(mDebugPS);
	ReleaseCOM(mDebugVS);

	ReleaseCOM(MatrixBuffer);
}

void MapClass::Update(float dt, ID3D11DeviceContext1 * mImmediateContext, std::shared_ptr<CameraClass> Camera)
{
	//Water->evaluateWavesGPU(dt, mImmediateContext);
	//WaterB->EvaluateWaves(dt, mImmediateContext);
	//WaterB->BEvelWater(dt, mImmediateContext);
	//Ocean->Update(mImmediateContext, dt, light, Camera);

	/*XMFLOAT3 dir_f = light.Direction();
	XMVECTOR dir = XMLoadFloat3(&dir_f);

	dir = XMVector3Transform(dir, XMMatrixRotationZ(dt * XM_2PI * (24*60) / (3600.0f * 24.0f)));

	XMStoreFloat3(&dir_f, dir);
	light.Direction(dir_f);*/
	light.SetLitWorld(XMFLOAT3(-768.0f, -150.0f, -768.0f), XMFLOAT3(768.0f, 150.0f, 768.0f));

	Clouds2->Update(dt);
}

void MapClass::Draw(ID3D11DeviceContext1 * mImmediateContext, std::shared_ptr<CameraClass> Camera)
{
	ShadowMap->BindDsvAndSetNullRenderTarget(mImmediateContext);
	ShadowMap->ClearDepthMap(mImmediateContext);

	Terrain->Draw(mImmediateContext, Camera, light, nullptr);

	RenderTargetStack::Pop(mImmediateContext);
	ViewportStack::Pop(mImmediateContext);

	Terrain->Draw(mImmediateContext, Camera, light, ShadowMap->DepthMapSRV());

	Sky->Draw(mImmediateContext, Camera, light);
}

void MapClass::Draw20(ID3D11DeviceContext1 * mImmediateContext, std::shared_ptr<CameraClass> Camera)
{
	
}

void MapClass::DrawDebug(ID3D11DeviceContext1 * mImmediateContext, std::shared_ptr<CameraClass> Camera)
{
	// CUBE
	UINT stride = sizeof(XMFLOAT4);
	UINT offset = 0;

	mImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	mImmediateContext->IASetVertexBuffers(0, 1, &mCubeVB, &stride, &offset);
	mImmediateContext->IASetIndexBuffer(mCubeIB, DXGI_FORMAT_R16_UINT, 0);
	mImmediateContext->IASetInputLayout(mCubeIL);

	MatrixBufferParams.gWorldProj = Camera->GetViewProjTransMatrix() * XMMatrixTranspose(XMMatrixTranslation(0.0f, 0.0f, 0.0)*XMMatrixScaling(100.0, 100.0, 100.0));
	MapResources(mImmediateContext, MatrixBuffer, MatrixBufferParams);

	mImmediateContext->VSSetShader(mCubeVS, nullptr, 0);
	mImmediateContext->VSSetConstantBuffers(0, 1, &MatrixBuffer);

	mImmediateContext->PSSetShader(mCubePS, nullptr, 0);
	
	mImmediateContext->RSSetState(RenderStates::Rasterizer::NoCullingRS);

	mImmediateContext->DrawIndexed(36, 0, 0);

}

float MapClass::GetHeight(float x, float y)
{
	return 0.3;
	return Terrain->GetHeight(x, y);
}