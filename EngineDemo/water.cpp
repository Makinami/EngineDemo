#include "water.h"

WaterClass::WaterClass() :
	mQuadPatchVB(nullptr),
	mQuadPatchIB(nullptr),
	MatrixBuffer(nullptr),
	mInputLayout(nullptr),
	mVertexShader(nullptr),
	mPixelShader(nullptr)
{
	XMStoreFloat4x4(&mWorld, XMMatrixIdentity());
}

WaterClass::~WaterClass()
{
	ReleaseCOM(mQuadPatchIB);
	ReleaseCOM(mQuadPatchVB);

	ReleaseCOM(MatrixBuffer);

	ReleaseCOM(mInputLayout);
	ReleaseCOM(mVertexShader);
	ReleaseCOM(mPixelShader);
}

bool WaterClass::Init(ID3D11Device1 * device, ID3D11DeviceContext1 * dc)
{
	BuildQuadPatchVB(device);
	BuildQuadPatchIB(device);

	if (!CreateInputLayoutAndShaders(device)) return false;

	// VS buffers
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

void WaterClass::Draw(ID3D11DeviceContext1 * mImmediateContext, std::shared_ptr<CameraClass> Camera)
{
	XMMATRIX ViewProjTrans = Camera->GetViewProjTransMatrix();

	UINT stride = sizeof(Vertex);
	UINT offset = 0;

	mImmediateContext->PSSetShader(mPixelShader, NULL, 0);

	D3D11_MAPPED_SUBRESOURCE mappedResources;
	MatrixBufferType *dataPtr;

	mImmediateContext->Map(MatrixBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResources);

	dataPtr = (MatrixBufferType*)mappedResources.pData;
	
	dataPtr->gWorld = ViewProjTrans*XMMatrixTranspose(XMMatrixTranslation(0.0f, 15.0f, 0.0f));

	mImmediateContext->Unmap(MatrixBuffer, 0);

	mImmediateContext->VSSetConstantBuffers(0, 1, &MatrixBuffer);

	mImmediateContext->VSSetShader(mVertexShader, NULL, 0);

	mImmediateContext->IASetInputLayout(mInputLayout);

	mImmediateContext->IASetIndexBuffer(mQuadPatchIB, DXGI_FORMAT_R16_UINT, 0);
	mImmediateContext->IASetVertexBuffers(0, 1, &mQuadPatchVB, &stride, &offset);
	mImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

	mImmediateContext->DrawIndexed(4, 0, 0);
}

void WaterClass::DrawWithShadow(ID3D11DeviceContext1 * mImmediateContext, std::shared_ptr<CameraClass> Camera, ID3D11ShaderResourceView * shadowmap)
{
	XMMATRIX ViewProjTrans = Camera->GetViewProjTransMatrix();

	UINT stride = sizeof(Vertex);
	UINT offset = 0;

	mImmediateContext->PSSetShader(mPixelShader, NULL, 0);
	mImmediateContext->PSSetShaderResources(0, 1, &shadowmap);

	D3D11_MAPPED_SUBRESOURCE mappedResources;
	MatrixBufferType *dataPtr;

	mImmediateContext->Map(MatrixBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResources);

	dataPtr = (MatrixBufferType*)mappedResources.pData;

	dataPtr->gWorld = ViewProjTrans*XMMatrixTranspose(XMMatrixTranslation(0.0f, 15.0f, 0.0f));

	mImmediateContext->Unmap(MatrixBuffer, 0);

	mImmediateContext->VSSetConstantBuffers(0, 1, &MatrixBuffer);

	mImmediateContext->VSSetShader(mVertexShader, NULL, 0);

	mImmediateContext->IASetInputLayout(mInputLayout);

	mImmediateContext->IASetIndexBuffer(mQuadPatchIB, DXGI_FORMAT_R16_UINT, 0);
	mImmediateContext->IASetVertexBuffers(0, 1, &mQuadPatchVB, &stride, &offset);
	mImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

	mImmediateContext->DrawIndexed(4, 0, 0);
}

void WaterClass::BuildQuadPatchVB(ID3D11Device1 * device)
{
	vector<Vertex> patchVertices;

	patchVertices.push_back(Vertex{ XMFLOAT3(-512.0f, 0.0f, 512.0f) });
	patchVertices.push_back(Vertex{ XMFLOAT3(512.0f, 0.0f, 512.0f) });
	patchVertices.push_back(Vertex{ XMFLOAT3(-512.0f, 0.0f, -512.0f) });
	patchVertices.push_back(Vertex{ XMFLOAT3(512.0f, 0.0f, -512.0f) });

	D3D11_BUFFER_DESC vbd;
	vbd.Usage = D3D11_USAGE_IMMUTABLE;
	vbd.ByteWidth = sizeof(Vertex)*patchVertices.size();
	vbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	vbd.CPUAccessFlags = 0;
	vbd.MiscFlags = 0;
	vbd.StructureByteStride = 0;

	D3D11_SUBRESOURCE_DATA vinitData;
	vinitData.pSysMem = &patchVertices[0];
	device->CreateBuffer(&vbd, &vinitData, &mQuadPatchVB);
}

bool WaterClass::BuildQuadPatchIB(ID3D11Device1 * device)
{
	vector<USHORT> indices(4);

	indices[0] = 0;
	indices[1] = 1;
	indices[2] = 2;
	indices[3] = 3;

	D3D11_BUFFER_DESC ibd;
	ibd.Usage = D3D11_USAGE_IMMUTABLE;
	ibd.ByteWidth = sizeof(USHORT)*indices.size();
	ibd.BindFlags = D3D11_BIND_INDEX_BUFFER;
	ibd.CPUAccessFlags = 0;
	ibd.MiscFlags = 0;
	ibd.StructureByteStride = 0;

	D3D11_SUBRESOURCE_DATA iinitData;
	iinitData.pSysMem = &indices[0];
	if (FAILED(device->CreateBuffer(&ibd, &iinitData, &mQuadPatchIB))) return false;

	return true;
}

bool WaterClass::CreateInputLayoutAndShaders(ID3D11Device1 * device)
{
	ifstream stream;
	size_t size;
	char* data;

	// pixel
	stream.open("..\\Debug\\WaterPS.cso", ifstream::binary);
	if (stream.good())
	{
		stream.seekg(0, ios::end);
		size = size_t(stream.tellg());
		data = new char[size];
		stream.seekg(0, ios::beg);
		stream.read(&data[0], size);
		stream.close();

		if (FAILED(device->CreatePixelShader(data, size, 0, &mPixelShader)))
		{
			LogError(L"Failed to create water pixel shader");
			return false;
		}
		delete[] data;
	}
	else
	{
		LogError(L"Failed to open WaterPS.cso");
		return false;
	}

	LogSuccess(L"Water pixel shader created.");

	// vertex shader
	stream.open("..\\Debug\\WaterVS.cso", ifstream::binary);
	if (stream.good())
	{
		stream.seekg(0, ios::end);
		size = size_t(stream.tellg());
		data = new char[size];
		stream.seekg(0, ios::beg);
		stream.read(&data[0], size);
		stream.close();

		if (FAILED(device->CreateVertexShader(data, size, 0, &mVertexShader)))
		{
			LogError(L"Failed to create water vertex shader");
			return false;
		}
	}
	else
	{
		LogError(L"Fail to open WaterPS.cso");
		return false;
	}

	LogSuccess(L"Water vertex shader created.");

	D3D11_INPUT_ELEMENT_DESC vertexDesc[] =
	{
		{"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0}
	};

	int numElements = sizeof(vertexDesc) / sizeof(vertexDesc[0]);

	if (FAILED(device->CreateInputLayout(vertexDesc, numElements, data, size, &mInputLayout))) return false;

	delete[] data;

	return true;
}
