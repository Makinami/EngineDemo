#include "map.h"

#include  "Utilities\RenderViewTargetStack.h"

MapClass::MapClass() :
	Terrain(nullptr),
	Water(nullptr)
{
}

MapClass::~MapClass()
{
	Shutdown();
}

bool MapClass::Init(ID3D11Device1* device, ID3D11DeviceContext1 * dc)
{
	Water = std::make_shared<WaterClass>();
	if (!Water->Init(device, dc))
	{
		LogError(L"Failed to initiate water");
		return false;
	}
	LogSuccess(L"Water initiated");

	Water20 = std::make_shared<WaterClass20>();
	Water20->Init(device, dc);

	Terrain = std::make_shared<TerrainClass>();
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

	ShadowMap = std::make_unique<ShadowMapClass>(device, 2048, 2048);

	light.Ambient(XMFLOAT4(0.2f, 0.2f, 0.2f, 1.0f));
	light.Diffuse(XMFLOAT4(0.5f, 0.5f, 0.5f, 1.0f));
	light.Specular(XMFLOAT4(0.5f, 0.5f, 0.5f, 1.0f));
	light.Direction(XMFLOAT3(0.994987f, -0.1f, 0.0f));

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

	ifstream stream;
	size_t size;;
	char* data;

	// pixel
	stream.open("..\\Debug\\DebugPS.cso", ifstream::in | ifstream::binary);
	if (stream.good())
	{
		stream.seekg(0, ios::end);
		size = size_t(stream.tellg());
		data = new char[size];
		stream.seekg(0, ios::beg);
		stream.read(&data[0], size);
		stream.close();

		if (FAILED(device->CreatePixelShader(data, size, 0, &mDebugPS)))
		{
			LogError(L"Failed to create Pixel Shader");
			return false;
		}
		delete[] data;
	}
	else
	{
		LogError(L"Failed to open TerrainPS.cso");
		return false;
	}

	LogSuccess(L"Pixel Shader created.");

	// vertex
	stream.open("..\\Debug\\DebugVS.cso", ifstream::in | ifstream::binary);
	if (stream.good())
	{
		stream.seekg(0, ios::end);
		size = size_t(stream.tellg());
		data = new char[size];
		stream.seekg(0, ios::beg);
		stream.read(&data[0], size);
		stream.close();

		if (FAILED(device->CreateVertexShader(data, size, 0, &mDebugVS)))
		{
			LogError(L"Failed to create Vertex Shader");
			return false;
		}
	}
	else
	{
		LogError(L"Failed to open TerrainVS.cso");
		return false;
	}

	LogSuccess(L"Vertex Shader created.");

	D3D11_INPUT_ELEMENT_DESC vertexDesc[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "TEXCOORD", 1, DXGI_FORMAT_R32G32_FLOAT, 0, 20, D3D11_INPUT_PER_VERTEX_DATA, 0 }
	};

	int numElements = sizeof(vertexDesc) / sizeof(vertexDesc[0]);

	if (FAILED(device->CreateInputLayout(vertexDesc, numElements, data, size, &mDebugIL))) return false;

	delete[] data;

	// DS
	D3D11_BUFFER_DESC matrixBufferDesc;
	matrixBufferDesc.Usage = D3D11_USAGE_DYNAMIC;
	matrixBufferDesc.ByteWidth = sizeof(MatrixBufferType);
	matrixBufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	matrixBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	matrixBufferDesc.MiscFlags = 0;
	matrixBufferDesc.StructureByteStride = 0;

	if (FAILED(device->CreateBuffer(&matrixBufferDesc, NULL, &MatrixBuffer))) return false;

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

void MapClass::Update(float dt, ID3D11DeviceContext1 * mImmediateContext)
{
	//Water->evaluateWavesFFT(dt);
	Water20->evaluateWavesGPU(dt, mImmediateContext);
	//Water->evaluateWavesGPU(dt, mImmediateContext);
}

void MapClass::Draw(ID3D11DeviceContext1 * mImmediateContext, std::shared_ptr<CameraClass> Camera)
{
	light.SetLitWorld(XMFLOAT3(-513.0f, 0.0f, -513.0f), XMFLOAT3(513.0f, 100.0f, 513.0f));

	/*ShadowMap->BindDsvAndSetNullRenderTarget(mImmediateContext);
	ShadowMap->ClearDepthMap(mImmediateContext);

	Terrain->Draw(mImmediateContext, Camera, light);

	RenderTargetStack::Pop(mImmediateContext);
	ViewportStack::Pop(mImmediateContext);

	//Terrain->Draw(mImmediateContext, Camera, light, ShadowMap->DepthMapSRV());

	//DrawDebug(mImmediateContext);
	*/
	//Water->Draw(mImmediateContext, Camera);
	Water20->Draw(mImmediateContext, Camera);
}

void MapClass::Draw20(ID3D11DeviceContext1 * mImmediateContext, std::shared_ptr<CameraClass> Camera)
{
	
}

void MapClass::DrawDebug(ID3D11DeviceContext1 * mImmediateContext)
{
	UINT stride = sizeof(TerrainClass::Vertex);
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
	mImmediateContext->PSSetShaderResources(0, 1, &mDepthMapSRV);

}

float MapClass::GetHeight(float x, float y)
{
	return Terrain->GetHeight(x, y);
}
