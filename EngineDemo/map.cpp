#include "map.h"

#include "Utilities\RenderViewTargetStack.h"
#include "Utilities\CreateShader.h"
#include "Utilities\MapResources.h"

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
	WaterB = std::make_unique<WaterBruneton>();
	WaterB->SetPerformance(Performance);
	if (!WaterB->Init(device, dc))
	{
		LogError(L"Failed to initiate water bruneton");
		return false;
	}
	LogSuccess(L"WaterBruneton initiated");

	/*Water = std::make_shared<WaterClass>();
	Water->SetPerformance(Performance);
	if (!Water->Init(device, dc))
	{
		LogError(L"Failed to initiate water");
		return false;
	}
	LogSuccess(L"Water initiated");*/

	Sky = std::make_shared<SkyClass>();
	Sky->SetPerformance(Performance);
	Sky->Init(device, dc);

	Ocean = std::make_unique<OceanClass>();
	if (FAILED(Ocean->Init(device, dc)))
	{
		LogError(L"Failed to initiate ocean");
		return false;
	}
	LogSuccess(L"Ocean initiated");

	/*Terrain = std::make_shared<TerrainClass>();
	Terrain->SetLogger(Logger);

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
	LogSuccess(L"Terrain initiated");
	*/
	Clouds = std::make_shared<CloudsClass>();
	//Clouds->Init(device, dc);

	Clouds2 = std::make_shared<CloudsClass2>();
	//Clouds2->Init(device, dc);

	ShadowMap = std::make_unique<ShadowMapClass>(device, 2048, 2048);

	light.Ambient(XMFLOAT4(0.2f, 0.2f, 0.2f, 1.0f));
	light.Diffuse(XMFLOAT4(0.5f, 0.5f, 0.5f, 1.0f));
	light.Specular(XMFLOAT4(0.5f, 0.5f, 0.5f, 1.0f));
	light.Direction(XMFLOAT3(-1.0f, 0.0f, 0.0f));

	vector<TerrainClass::Vertex> patchVertices(4);

	patchVertices[0] = TerrainClass::Vertex{ XMFLOAT3(-1.0f, -1.0f, 0.0f), XMFLOAT2(0.0f, 1.0f) };
	patchVertices[1] = TerrainClass::Vertex{ XMFLOAT3(-1.0f, 1.0f, 0.0f), XMFLOAT2(0.0f, 0.0f) };
	patchVertices[2] = TerrainClass::Vertex{ XMFLOAT3(1.0f, 1.0f, 0.0f), XMFLOAT2(1.0f, 0.0f) };
	patchVertices[3] = TerrainClass::Vertex{ XMFLOAT3(1.0f, -1.0f, 0.0f), XMFLOAT2(1.0f, 1.0f) };


	D3D11_BUFFER_DESC vbd;
	vbd.Usage = D3D11_USAGE_IMMUTABLE;
	vbd.ByteWidth = sizeof(TerrainClass::Vertex)*patchVertices.size();
	vbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	vbd.CPUAccessFlags = 0;
	vbd.MiscFlags = 0;
	vbd.StructureByteStride = 0;

	D3D11_SUBRESOURCE_DATA vinitData;
	vinitData.pSysMem = &patchVertices[0];
	device->CreateBuffer(&vbd, &vinitData, &mScreenQuadVB);

	vector<USHORT> indices(6);
	
	indices[0] = 0;
	indices[1] = 1;
	indices[2] = 2;
	indices[3] = 0;
	indices[4] = 2;
	indices[5] = 3;

	D3D11_BUFFER_DESC ibd;
	ibd.Usage = D3D11_USAGE_IMMUTABLE;
	ibd.ByteWidth = sizeof(USHORT)*indices.size();
	ibd.BindFlags = D3D11_BIND_INDEX_BUFFER;
	ibd.CPUAccessFlags = 0;
	ibd.MiscFlags = 0;
	ibd.StructureByteStride = 0;

	D3D11_SUBRESOURCE_DATA iinitData;
	iinitData.pSysMem = &indices[0];
	if (FAILED(device->CreateBuffer(&ibd, &iinitData, &mScreenQuadIB))) return false;

	// pixel
	CreatePSFromFile(L"..\\Debug\\DebugPS.cso", device, mDebugPS);
	
	// vertex
	D3D11_INPUT_ELEMENT_DESC vertexDesc[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "TEXCOORD", 1, DXGI_FORMAT_R32G32_FLOAT, 0, 20, D3D11_INPUT_PER_VERTEX_DATA, 0 }
	};

	int numElements = sizeof(vertexDesc) / sizeof(vertexDesc[0]);

	CreateVSAndInputLayout(L"..\\Debug\\DebugVS.cso", device, mDebugVS, vertexDesc, numElements, mDebugIL);

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
	vbd.ByteWidth = sizeof(cubeVertices[0])*cubeVertices.size();
	vinitData.pSysMem = &cubeVertices[0];
	device->CreateBuffer(&vbd, &vinitData, &mCubeVB);

	vector<USHORT> cubeIndices{ 0,1,2,1,2,3,4,5,6,5,6,7,8,9,10,9,10,11,12,13,14,13,14,15,16,17,18,17,18,19,20,21,22,21,22,23 };
	ibd.ByteWidth = sizeof(cubeIndices[0])*cubeIndices.size();
	iinitData.pSysMem = &cubeIndices[0];
	device->CreateBuffer(&ibd, &iinitData, &mCubeIB);

	CreatePSFromFile(L"..\\Debug\\cubePS.cso", device, mCubePS);

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
	WaterB->EvaluateWaves(dt, mImmediateContext);
	//WaterB->BEvelWater(dt, mImmediateContext);
	Ocean->Update(mImmediateContext, dt, Camera);

	XMFLOAT3 dir_f = light.Direction();
	XMVECTOR dir = XMLoadFloat3(&dir_f);

	dir = XMVector3Transform(dir, XMMatrixRotationZ(dt*XM_2PI/(60.f*5.0f)));

	XMStoreFloat3(&dir_f, dir);
	light.Direction(dir_f);
}

void MapClass::Draw(ID3D11DeviceContext1 * mImmediateContext, std::shared_ptr<CameraClass> Camera)
{
	/*light.SetLitWorld(XMFLOAT3(-768.0f, -150.0f, -768.0f), XMFLOAT3(768.0f, 150.0f, 768.0f));

	ShadowMap->BindDsvAndSetNullRenderTarget(mImmediateContext);
	ShadowMap->ClearDepthMap(mImmediateContext);

	Terrain->Draw(mImmediateContext, Camera, light);

	//Clouds->GenerateClouds(mImmediateContext);

	RenderTargetStack::Pop(mImmediateContext);
	ViewportStack::Pop(mImmediateContext);
	*/
	//Terrain->Draw(mImmediateContext, Camera, light, ShadowMap->DepthMapSRV());

	//DrawDebug(mImmediateContext);
	
	//Water->Draw(mImmediateContext, Camera, light, ShadowMap->DepthMapSRV());
	//Water->Draw(mImmediateContext, Camera, light, WaterB->getFFTWaves());
	
	Sky->DrawToMap(mImmediateContext, light);
	Sky->DrawToCube(mImmediateContext, light);
	Sky->DrawToScreen(mImmediateContext, Camera, light);
	//Sky->Draw(mImmediateContext, Camera, light);

	WaterB->Draw(mImmediateContext, Camera, light);
	//Water->Draw(mImmediateContext, Camera, light, ShadowMap->DepthMapSRV());
	//Ocean->Draw(mImmediateContext, Camera, light);

	DrawDebug(mImmediateContext, Camera);
	
	/*static int counter = 0;

	if (counter == 0)
		Sky->DrawToCube(mImmediateContext, light);

	if (++counter == 1) counter = 0;
	
	Sky->DrawToScreen(mImmediateContext, Camera, light);*/

	//Clouds->Draw(mImmediateContext, Camera, light);
	//Clouds2->GenerateClouds(mImmediateContext);
	//Clouds2->Draw(mImmediacoteContext, Camera, light, Sky->getTransmittanceSRV());
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

	MatrixBufferParams.gWorldProj = Camera->GetViewProjTransMatrix() * XMMatrixTranspose(XMMatrixTranslation(392.0f, 0.0, 0.0));
	MapResources(mImmediateContext, MatrixBuffer, MatrixBufferParams);

	mImmediateContext->VSSetShader(mCubeVS, nullptr, 0);
	mImmediateContext->VSSetConstantBuffers(0, 1, &MatrixBuffer);

	mImmediateContext->PSSetShader(mCubePS, nullptr, 0);

	mImmediateContext->DrawIndexed(36, 0, 0);

	// SHADOW MAP
	/*UINT stride = sizeof(TerrainClass::Vertex);
	UINT offset = 0;

	mImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	mImmediateContext->IASetVertexBuffers(0, 1, &mScreenQuadVB, &stride, &offset);
	mImmediateContext->IASetIndexBuffer(mScreenQuadIB, DXGI_FORMAT_R16_UINT, 0);

	// Scale and shift quad to lower-right corner.
	XMMATRIX world(
	0.25f, 0.0f, 0.0f, 0.0f,
	0.0f, 0.25f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0f,
	0.75f, -0.75f, 0.0f, 1.0f);

	//PS
	ID3D11ShaderResourceView* mDepthMapSRV = ShadowMap->DepthMapSRV();
	mImmediateContext->PSSetShaderResources(0, 1, &mDepthMapSRV);
	mImmediateContext->PSSetShader(mDebugPS, NULL, 0);

	D3D11_MAPPED_SUBRESOURCE mappedResources;
	MatrixBufferType *dataPtr;

	mImmediateContext->Map(MatrixBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResources);

	dataPtr = (MatrixBufferType*)mappedResources.pData;

	dataPtr->gWorldProj = XMMatrixTranspose(world);

	mImmediateContext->Unmap(MatrixBuffer, 0);

	mImmediateContext->VSSetConstantBuffers(0, 1, &MatrixBuffer);

	mImmediateContext->VSSetShader(mDebugVS, NULL, 0);

	mImmediateContext->IASetInputLayout(mDebugIL);

	mImmediateContext->DrawIndexed(6, 0, 0);

	mDepthMapSRV = NULL;
	mImmediateContext->PSSetShaderResources(0, 1, &mDepthMapSRV);*/

}

float MapClass::GetHeight(float x, float y)
{
	return 0.3;
	return Terrain->GetHeight(x, y);
}