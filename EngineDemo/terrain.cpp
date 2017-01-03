#include "terrain.h"
#include "DDSTextureLoader.h"
#include "Utilities\CreateShader.h"

#include "RenderStates.h"

TerrainClass::TerrainClass() :
	mQuadPatchVB(0),
	mQuadPatchIB(0),
	cbPerFrameHS(0),
	mHeightmapSRV(0),
	mNumPatchVertices(0),
	mNumPatchQuadFaces(0),
	mNumPatchVertRows(0),
	mNumPatchVertCols(0),
	MatrixBuffer(0),
	cbPerFramePS(0),
	mLayerMapArraySRV(0),
	mBlendMapSRV(0),
	mInputLayout(0),
	mVertexShader(0),
	mHullShader(0),
	mDomainShader(0),
	mPixelShader(0),
	mRastStateBasic(0),
	mRastStateShadow(0)
{
	XMStoreFloat4x4(&mWorld, XMMatrixIdentity());
}

TerrainClass::~TerrainClass()
{
	ReleaseCOM(mQuadPatchIB);
	ReleaseCOM(mQuadPatchVB);

	ReleaseCOM(cbPerFrameHS);
	ReleaseCOM(MatrixBuffer);
	ReleaseCOM(cbPerFramePS);

	ReleaseCOM(mLayerMapArraySRV);
	ReleaseCOM(mBlendMapSRV);
	ReleaseCOM(mHeightmapSRV);

	ReleaseCOM(mInputLayout);
	ReleaseCOM(mVertexShader);
	ReleaseCOM(mHullShader);
	ReleaseCOM(mDomainShader);
	ReleaseCOM(mPixelShader);

	ReleaseCOM(mRastStateBasic);
	ReleaseCOM(mRastStateShadow);
	//ReleaseCOM(mSamplerStates[0]);
	//ReleaseCOM(mSamplerStates[1]);
	//ReleaseCOM(mSamplerStates[2]);
	//delete[] mSamplerStates;
}

float TerrainClass::GetWidth() const
{
	return (mInfo.HeightmapWidth - 1)*mInfo.CellSpacing;
}

float TerrainClass::GetDepth() const
{
	return (mInfo.HeightmapHeight - 1)*mInfo.CellSpacing;
}

float TerrainClass::GetHeight(float x, float y) const
{
	float c = (x + 0.5f*GetWidth()) / mInfo.CellSpacing;
	float d = (y - 0.5f*GetDepth()) / -mInfo.CellSpacing;

	int row = (int)floorf(d);
	int col = (int)floorf(c);

	float A = mHeightmap[row*mInfo.HeightmapWidth + col];
	float B = mHeightmap[row*mInfo.HeightmapWidth + col + 1];
	float C = mHeightmap[(row + 1)*mInfo.HeightmapWidth + col];
	float D = mHeightmap[(row + 1)*mInfo.HeightmapWidth + col + 1];

	float s = c - (float)col;
	float t = d - (float)row;

	if (s + t <= 1.0f)
	{
		float uy = B - A;
		float vy = C - A;
		return A + s*uy + t*vy;
	}
	else
	{
		float uy = C - D;
		float vy = B - D;
		return D + (1.0f - s)*uy + (1.0f - t)*vy;
	}
}

XMMATRIX TerrainClass::GetWorld() const
{
	return XMLoadFloat4x4(&mWorld);
}

void TerrainClass::SetWorld(CXMMATRIX M)
{
	XMStoreFloat4x4(&mWorld, M);
}

bool TerrainClass::Init(ID3D11Device1* device, ID3D11DeviceContext1* dc, const InitInfo& initInfo)
{
	mInfo = initInfo;

	mNumPatchVertRows = ((mInfo.HeightmapHeight - 1) / CellsPerPatch) + 1;
	mNumPatchVertCols = ((mInfo.HeightmapWidth - 1) / CellsPerPatch) + 1;

	mNumPatchVertices = mNumPatchVertRows*mNumPatchVertCols;
	mNumPatchQuadFaces = (mNumPatchVertRows - 1)*(mNumPatchVertCols - 1);
	
	LoadHeighMap();
	//Smooth();
	CalcAllPatchBoundsY();
	
	BuildQuadPatchVB(device);
	if (!BuildQuadPatchIB(device)) return false;
	BuildHeightmapSRV(device);

	if (!CreateInputLayoutAndShaders(device)) return false;

	std::vector<std::wstring> layerFilenames;
	layerFilenames.push_back(mInfo.LayerMapFilename0);
	layerFilenames.push_back(mInfo.LayerMapFilename1);
	layerFilenames.push_back(mInfo.LayerMapFilename2);
	layerFilenames.push_back(mInfo.LayerMapFilename3);
	layerFilenames.push_back(mInfo.LayerMapFilename4);
	mLayerMapArraySRV = CreateTexture2DArraySRV(device, dc, layerFilenames);
	
	CreateDDSTextureFromFile(device, mInfo.BlendMapFilename.c_str(), NULL, &mBlendMapSRV);	

	// DS
	D3D11_BUFFER_DESC matrixBufferDesc;
	matrixBufferDesc.Usage = D3D11_USAGE_DYNAMIC;
	matrixBufferDesc.ByteWidth = sizeof(MatrixBufferType);
	matrixBufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	matrixBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	matrixBufferDesc.MiscFlags = 0;
	matrixBufferDesc.StructureByteStride = 0;

	if (FAILED(device->CreateBuffer(&matrixBufferDesc, NULL, &MatrixBuffer))) return false;

	// HS
	D3D11_BUFFER_DESC cbPerFrameHSDesc;
	cbPerFrameHSDesc.Usage = D3D11_USAGE_DYNAMIC;
	cbPerFrameHSDesc.ByteWidth = sizeof(cbPerFrameHSType);
	cbPerFrameHSDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	cbPerFrameHSDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	cbPerFrameHSDesc.MiscFlags = 0;
	cbPerFrameHSDesc.StructureByteStride = 0;

	if (FAILED(device->CreateBuffer(&cbPerFrameHSDesc, NULL, &cbPerFrameHS))) return false;
	
	// PS
	D3D11_BUFFER_DESC cbPerFramePSDesc;
	cbPerFramePSDesc.Usage = D3D11_USAGE_DYNAMIC;
	cbPerFramePSDesc.ByteWidth = sizeof(cbPerFramePSType);
	cbPerFramePSDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	cbPerFramePSDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	cbPerFramePSDesc.MiscFlags = 0;
	cbPerFramePSDesc.StructureByteStride = 0;

	if (FAILED(device->CreateBuffer(&cbPerFramePSDesc, NULL, &cbPerFramePS))) return false;

	D3D11_RASTERIZER_DESC rastDesc;
	ZeroMemory(&rastDesc, sizeof(D3D11_RASTERIZER_DESC));
	rastDesc.FillMode = D3D11_FILL_SOLID;
	rastDesc.CullMode = D3D11_CULL_BACK;
	rastDesc.FrontCounterClockwise = false;
	rastDesc.DepthClipEnable = true;

	if (FAILED(device->CreateRasterizerState(&rastDesc, &mRastStateBasic))) return false;

	rastDesc.DepthBias = 100000;
	rastDesc.DepthBiasClamp = 0.0f;
	rastDesc.SlopeScaledDepthBias = 1.0f;

	if (FAILED(device->CreateRasterizerState(&rastDesc, &mRastStateShadow))) return false;

	mSamplerStates = new ID3D11SamplerState*[3];

	mSamplerStates[0] = RenderStates::Sampler::BilinearClampSS;
	mSamplerStates[1] = RenderStates::Sampler::TrilinearWrapSS;
	mSamplerStates[2] = RenderStates::Sampler::BilinearClampComLessSS;

	return true;
}

void TerrainClass::Draw(ID3D11DeviceContext1 * mImmediateContext, std::shared_ptr<CameraClass> Camera, DirectionalLight& light, ID3D11ShaderResourceView * ShadowMap)
{
	XMMATRIX ShadowViewProjTrans = light.GetViewProjTrans();//XMMatrixTranspose(V*P);
	XMMATRIX ShadowMapProjTrans = light.GetMapProjTrans();//XMMatrixTranspose(V*P*T);

	XMMATRIX ViewProjTrans = Camera->GetViewProjTransMatrix();

	UINT stride = sizeof(Vertex);
	UINT offset = 0;

	XMFLOAT4 worldPlanes[6];
	ExtractFrustrumPlanes(worldPlanes, Camera->GetViewProjMatrix());

	// PS
	if (ShadowMap)
	{
		D3D11_MAPPED_SUBRESOURCE mappedResourcePS;
		cbPerFramePSType* dataPtrPS;

		mImmediateContext->Map(cbPerFramePS, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResourcePS);
		dataPtrPS = (cbPerFramePSType*)mappedResourcePS.pData;

		XMStoreFloat3(&(dataPtrPS->gEyePosW), Camera->GetPosition());
		dataPtrPS->gTexelCellSpaceU = 1.0f / mInfo.HeightmapWidth;
		dataPtrPS->gTexelCellSpaceV = 1.0f / mInfo.HeightmapHeight;
		dataPtrPS->gWorldCellSpace = mInfo.CellSpacing;
		dataPtrPS->gDirLight = light.LightParams();
		dataPtrPS->gViewProj = Camera->GetViewProjTransMatrix();

		mImmediateContext->Unmap(cbPerFramePS, 0);

		mImmediateContext->PSSetConstantBuffers(0, 1, &cbPerFramePS);

		mImmediateContext->PSSetShaderResources(0, 1, &mHeightmapSRV);
		mImmediateContext->PSSetShaderResources(1, 1, &mBlendMapSRV);
		mImmediateContext->PSSetShaderResources(2, 1, &mLayerMapArraySRV);
		mImmediateContext->PSSetShaderResources(3, 1, &ShadowMap);
		mImmediateContext->PSSetSamplers(0, 3, mSamplerStates);

		mImmediateContext->PSSetShader(mPixelShader, NULL, 0);
	}
	else
	{
		mImmediateContext->PSSetShader(NULL, NULL, 0);
	}

	// DS
	D3D11_MAPPED_SUBRESOURCE mappedResourceDS;
	MatrixBufferType* dataPtrDS;

	mImmediateContext->Map(MatrixBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResourceDS);
	dataPtrDS = (MatrixBufferType*)mappedResourceDS.pData;

	if (ShadowMap) dataPtrDS->gWorldProj = Camera->GetViewProjTransMatrix();
	else dataPtrDS->gWorldProj = ShadowViewProjTrans;
	dataPtrDS->gShadowTrans = ShadowMapProjTrans;

	mImmediateContext->Unmap(MatrixBuffer, 0);

	mImmediateContext->DSSetConstantBuffers(0, 1, &MatrixBuffer);
	mImmediateContext->DSSetShaderResources(0, 1, &mHeightmapSRV);

	mImmediateContext->DSSetSamplers(0, 1, &mSamplerStates[0]);

	mImmediateContext->DSSetShader(mDomainShader, NULL, 0);

	// HS
	D3D11_MAPPED_SUBRESOURCE mappedResourceHS;
	cbPerFrameHSType* dataPtrHS;

	mImmediateContext->Map(cbPerFrameHS, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResourceHS);
	dataPtrHS = (cbPerFrameHSType*)mappedResourceHS.pData;

	XMStoreFloat3(&(dataPtrHS->gEyePosW), Camera->GetPosition());
	dataPtrHS->gMaxDist = 500.0f;
	dataPtrHS->gMinDist = 20.0f;
	dataPtrHS->gMaxTess = 6.0f;
	dataPtrHS->gMinTess = 0.0f;
	for (int i = 0; i < 6; ++i)
		dataPtrHS->gWorldFrustumPlanes[i] = worldPlanes[i];
	if (ShadowMap) dataPtrHS->gFrustumCull = 1;
	else dataPtrHS->gFrustumCull = 0;

	mImmediateContext->Unmap(cbPerFrameHS, 0);

	mImmediateContext->HSSetConstantBuffers(0, 1, &cbPerFrameHS);

	mImmediateContext->HSSetShader(mHullShader, NULL, 0);

	// VS
	mImmediateContext->VSSetShaderResources(0, 1, &mHeightmapSRV);

	mImmediateContext->VSSetSamplers(0, 1, &mSamplerStates[0]);

	mImmediateContext->VSSetShader(mVertexShader, NULL, 0);

	// IA
	mImmediateContext->IASetIndexBuffer(mQuadPatchIB, DXGI_FORMAT_R16_UINT, 0);
	mImmediateContext->IASetVertexBuffers(0, 1, &mQuadPatchVB, &stride, &offset);
	mImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_4_CONTROL_POINT_PATCHLIST);

	mImmediateContext->IASetInputLayout(mInputLayout);

	if (ShadowMap) mImmediateContext->RSSetState(mRastStateBasic);
	else mImmediateContext->RSSetState(mRastStateShadow);

	mImmediateContext->DrawIndexed(mNumPatchQuadFaces * 4, 0, 0);

	mImmediateContext->HSSetShader(0, 0, 0);
	mImmediateContext->DSSetShader(0, 0, 0);

	ID3D11ShaderResourceView* NULLRes = NULL;
	mImmediateContext->PSSetShaderResources(0, 1, &NULLRes);
	mImmediateContext->PSSetShaderResources(3, 1, &NULLRes);
}

void TerrainClass::LoadHeighMap()
{
	// A height fo reach vertex
	std::vector<unsigned char> in(mInfo.HeightmapWidth*mInfo.HeightmapHeight);

	// Open the file.
	ifstream inFile;
	inFile.open(mInfo.HeightMapFilename.c_str(), ios_base::binary);

	if (inFile)
	{
		// Read the RAW bytes.
		inFile.read((char*)&in[0], (streamsize)in.size());

		// Done with the file.
		inFile.close();
	}

	// Copy the array data into a float array and scale it.
	mHeightmap.resize(mInfo.HeightmapHeight*mInfo.HeightmapWidth, 0);
	for (UINT i = 0; i < mInfo.HeightmapHeight*mInfo.HeightmapWidth; ++i)
		mHeightmap[i] = (in[i] / 255.0f)*mInfo.HeightScale;
}

void TerrainClass::Smooth()
{
	vector<float> dest(mHeightmap.size());

	for (UINT i = 0; i < mInfo.HeightmapHeight; ++i)
		for (UINT j = 0; j < mInfo.HeightmapWidth; ++j)
			dest[i*mInfo.HeightmapWidth + j] = Avarage(i, j);

	mHeightmap = dest;
}

bool TerrainClass::InBounds(int i, int j)
{
	// True if ij are valid indices; false otherwise.
	return
		i >= 0 && i < (int)mInfo.HeightmapHeight &&
		j >= 0 && j < (int)mInfo.HeightmapWidth;
}

float TerrainClass::Avarage(int i, int j)
{
	float avg = 0.0f;
	float num = 0.0f;

	for (int m = i - 1; m <= i + 1; ++m)
	{
		for (int n = j - 1; n <= j + 1; ++n)
		{
			if (InBounds(m, n))
			{
				avg += mHeightmap[m*mInfo.HeightmapWidth + n];
				++num;
			}
		}
	}

	return avg / num;
}

void TerrainClass::CalcAllPatchBoundsY()
{
	mPatchBoundsY.resize(mNumPatchQuadFaces);

	for (UINT i = 0; i < mNumPatchVertRows - 1; ++i)
	{
		for (UINT j = 0; j < mNumPatchVertCols - 1; ++j)
		{
			CalcPatchBoundsY(i, j);
		}
	}
}

void TerrainClass::CalcPatchBoundsY(UINT i, UINT j)
{
	UINT x0 = j*CellsPerPatch;
	UINT x1 = (j + 1)*CellsPerPatch;

	UINT y0 = i*CellsPerPatch;
	UINT y1 = (i + 1)*CellsPerPatch;

	float minY = +FLT_MAX;
	float maxY = -FLT_MAX;
	for (UINT x = x0; x < x1; ++x)
	{
		for (UINT y = y0; y < y1; ++y)
		{
			UINT k = y*mInfo.HeightmapWidth + x;
			minY = min(minY, mHeightmap[k]);
			maxY = max(maxY, mHeightmap[k]);
		}
	}

	mPatchBoundsY[i*(mNumPatchVertCols - 1) + j] = XMFLOAT2(minY, maxY);
}

void TerrainClass::BuildQuadPatchVB(ID3D11Device1* device)
{
	vector<Vertex> patchVertices(mNumPatchVertRows*mNumPatchVertCols);

	float halfWidth = 0.5f*GetWidth();
	float halfDepth = 0.5f*GetDepth();

	float patchWidth = GetWidth() / (mNumPatchVertCols - 1);
	float patchDepth = GetDepth() / (mNumPatchVertRows - 1);
	float du = 1.0f / (mNumPatchVertCols - 1);
	float dv = 1.0f / (mNumPatchVertRows - 1);
	
	for (UINT i = 0; i < mNumPatchVertRows; ++i)
	{
		float z = halfDepth - i*patchDepth;
		for (UINT j = 0; j < mNumPatchVertCols; ++j)
		{
			float x = -halfWidth + j*patchWidth;

			patchVertices[i*mNumPatchVertCols + j].Pos = XMFLOAT3(x, 0.0f, z);

			// Stretch texture over grid;
			patchVertices[i*mNumPatchVertCols + j].Tex.x = j*du;
			patchVertices[i*mNumPatchVertCols + j].Tex.y = i*dv;
		}
	}

	for (UINT i = 0; i < mNumPatchVertRows - 1; ++i)
		for (UINT j = 0; j < mNumPatchVertCols - 1; ++j)
			patchVertices[i*mNumPatchVertCols + j].BoundsY = mPatchBoundsY[i*(mNumPatchVertCols - 1) + j];

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

bool TerrainClass::BuildQuadPatchIB(ID3D11Device1* device)
{
	vector<USHORT> indices(mNumPatchQuadFaces * 4);

	int k = 0;
	for (UINT i = 0; i < mNumPatchVertRows - 1; ++i)
	{
		for (UINT j = 0; j < mNumPatchVertCols - 1; ++j)
		{
			indices[k] = i*mNumPatchVertCols + j;
			indices[k + 1] = i*mNumPatchVertCols + j + 1;
			indices[k + 2] = (i + 1)*mNumPatchVertCols + j;
			indices[k + 3] = (i + 1)*mNumPatchVertCols + j + 1;

			k += 4;
		}
	}

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

void TerrainClass::BuildHeightmapSRV(ID3D11Device1* device)
{
	D3D11_TEXTURE2D_DESC texDesc;
	texDesc.Width = mInfo.HeightmapWidth;
	texDesc.Height = mInfo.HeightmapHeight;
	texDesc.MipLevels = 1;
	texDesc.ArraySize = 1;
	texDesc.Format = DXGI_FORMAT_R16_FLOAT;
	texDesc.SampleDesc.Count = 1;
	texDesc.SampleDesc.Quality = 0;
	texDesc.Usage = D3D11_USAGE_DEFAULT;
	texDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	texDesc.CPUAccessFlags = 0;
	texDesc.MiscFlags = 0;

	vector<PackedVector::HALF> hmap(mHeightmap.size());
	transform(mHeightmap.begin(), mHeightmap.end(), hmap.begin(), PackedVector::XMConvertFloatToHalf);

	/*
	vector< vector<XMFLOAT3> > vNormals[2];
	for (int i = 0; i < 2; ++i) vNormals[i] = vector< vector<XMFLOAT3> >(mInfo.HeightmapWidth - 1, vector<XMFLOAT3>(mInfo.HeightmapHeight - 1));

	for (int row = 0; row < mInfo.HeightmapHeight-1; ++row)
	{
	for (int col = 0; col < mInfo.HeightmapWidth-1; ++col)
	{
	float A = mHeightmap[row*mInfo.HeightmapWidth + col];
	float B = mHeightmap[row*mInfo.HeightmapWidth + col + 1];
	float C = mHeightmap[(row + 1)*mInfo.HeightmapWidth + col];
	float D = mHeightmap[(row + 1)*mInfo.HeightmapWidth + col + 1];

	XMFLOAT3 AC = { 0.0f, C - A, 1.0f },
	AB = { 1.0f, B - A, 0.0f },
	DB = { 0.0f, B - D, -1.0f },
	DC = { -1.0f, C - D, 0.0f };

	XMStoreFloat3(&vNormals[0][row][col], XMVector3Normalize(XMVector3Cross(XMLoadFloat3(&AC), XMLoadFloat3(&AB))));
	XMStoreFloat3(&vNormals[1][row][col], XMVector3Normalize(XMVector3Cross(XMLoadFloat3(&DB), XMLoadFloat3(&DC))));
	}
	}

	vector<PackedVector::XMHALF4> hmap(mHeightmap.size());
	XMFLOAT3 Normal;
	for (int row = 0; row < mInfo.HeightmapHeight; ++row)
	{
	for (int col = 0; col < mInfo.HeightmapWidth; ++col)
	{
	Normal = { 0.0f, 0.0f, 0.0f };

	if (row && col) XMStoreFloat3(&Normal, XMLoadFloat3(&Normal) + XMLoadFloat3(&vNormals[1][row - 1][col - 1]));

	if (row && col < mInfo.HeightmapWidth-1)
	for (int i = 0; i < 2; ++i)
	XMStoreFloat3(&Normal, XMLoadFloat3(&Normal) + XMLoadFloat3(&vNormals[1][row - 1][col]));

	if (row < mInfo.HeightmapHeight-1 && col)
	for (int i = 0; i < 2; ++i)
	XMStoreFloat3(&Normal, XMLoadFloat3(&Normal) + XMLoadFloat3(&vNormals[1][row][col-1]));

	if (row < mInfo.HeightmapHeight-1 && col < mInfo.HeightmapWidth) XMStoreFloat3(&Normal, XMLoadFloat3(&Normal) + XMLoadFloat3(&vNormals[1][row][col]));

	XMStoreFloat3(&Normal, XMVector3Normalize(XMLoadFloat3(&Normal)));

	hmap[row*mInfo.HeightmapWidth + col] = { PackedVector::XMConvertFloatToHalf(Normal.x),
	PackedVector::XMConvertFloatToHalf(Normal.y),
	PackedVector::XMConvertFloatToHalf(Normal.z),
	PackedVector::XMConvertFloatToHalf(mHeightmap[row*mInfo.HeightmapWidth + col]) };
	}
	}
	*/

	D3D11_SUBRESOURCE_DATA data;
	data.pSysMem = &hmap[0];
	data.SysMemPitch = mInfo.HeightmapWidth*sizeof(PackedVector::HALF);
	data.SysMemSlicePitch = 0;

	ID3D11Texture2D* hmapTex = 0;
	device->CreateTexture2D(&texDesc, &data, &hmapTex);

	D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
	srvDesc.Format = texDesc.Format;
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Texture2D.MostDetailedMip = 0;
	srvDesc.Texture2D.MipLevels = -1;
	device->CreateShaderResourceView(hmapTex, &srvDesc, &mHeightmapSRV);

	ReleaseCOM(hmapTex);
}

bool TerrainClass::CreateInputLayoutAndShaders(ID3D11Device1* device)
{
	// pixel
	CreatePSFromFile(L"..\\Debug\\Shaders\\Terrain\\TerrainPS.cso", device, mPixelShader);

	// domain
	CreateDSFromFile(L"..\\Debug\\Shaders\\Terrain\\TerrainDS.cso", device, mDomainShader);

	// hull
	CreateHSFromFile(L"..\\Debug\\Shaders\\Terrain\\TerrainHS.cso", device, mHullShader);

	// vertex and input layout	
	D3D11_INPUT_ELEMENT_DESC vertexDesc[] =
	{
		{"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
		{"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0},
		{"TEXCOORD", 1, DXGI_FORMAT_R32G32_FLOAT, 0, 20, D3D11_INPUT_PER_VERTEX_DATA, 0}
	};

	int numElements = sizeof(vertexDesc) / sizeof(vertexDesc[0]);

	CreateVSAndInputLayout(L"..\\Debug\\Shaders\\Terrain\\TerrainVS.cso", device, mVertexShader, vertexDesc, numElements, mInputLayout);
	
	return true;
}

// temp
void ExtractFrustrumPlanes(XMFLOAT4 planes[6], CXMMATRIX M)
{
	XMFLOAT4X4 matrix;
	XMStoreFloat4x4(&matrix, M);

	// Left
	planes[0].x = matrix._14 + matrix._11;
	planes[0].y = matrix._24 + matrix._21;
	planes[0].z = matrix._34 + matrix._31;
	planes[0].w = matrix._44 + matrix._41;

	// Right
	planes[1].x = matrix._14 - matrix._11;
	planes[1].y = matrix._24 - matrix._21;
	planes[1].z = matrix._34 - matrix._31;
	planes[1].w = matrix._44 - matrix._41;

	// Bottom
	planes[2].x = matrix._14 + matrix._12;
	planes[2].y = matrix._24 + matrix._22;
	planes[2].z = matrix._34 + matrix._32;
	planes[2].w = matrix._44 + matrix._42;

	// Top
	planes[3].x = matrix._14 - matrix._12;
	planes[3].y = matrix._24 - matrix._22;
	planes[3].z = matrix._34 - matrix._32;
	planes[3].w = matrix._44 - matrix._42;

	// Near
	planes[4].x = matrix._13;
	planes[4].y = matrix._23;
	planes[4].z = matrix._33;
	planes[4].w = matrix._43;

	// Left
	planes[5].x = matrix._14 - matrix._13;
	planes[5].y = matrix._24 - matrix._23;
	planes[5].z = matrix._34 - matrix._33;
	planes[5].w = matrix._44 - matrix._43;

	// Normalize the plane equations.
	for (int i = 0; i < 6; ++i)
	{
		XMVECTOR v = XMPlaneNormalize(XMLoadFloat4(&planes[i]));
		XMStoreFloat4(&planes[i], v);
	}
}

ID3D11ShaderResourceView* CreateTexture2DArraySRV(ID3D11Device1* device, ID3D11DeviceContext1* context,	std::vector<std::wstring>& filenames)
{
	//
	// Load the texture elements individually from file.  These textures
	// won't be used by the GPU (0 bind flags), they are just used to 
	// load the image data from file.  We use the STAGING usage so the
	// CPU can read the resource.
	//

	UINT size = filenames.size();

	std::vector<ID3D11Texture2D*> srcTex(size);
	for (UINT i = 0; i < size; ++i)
	{
		CreateDDSTextureFromFileEx(device, filenames[i].c_str(), 0, D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE, 0, false, (ID3D11Resource**)&srcTex[i], NULL);
	}

	//
	// Create the texture array.  Each element in the texture 
	// array has the same format/dimensions.
	//
	
	D3D11_TEXTURE2D_DESC texElementDesc;
	srcTex[0]->GetDesc(&texElementDesc);

	D3D11_TEXTURE2D_DESC texArrayDesc;
	texArrayDesc.Width = texElementDesc.Width;
	texArrayDesc.Height = texElementDesc.Height;
	texArrayDesc.MipLevels = texElementDesc.MipLevels;
	texArrayDesc.ArraySize = size;
	texArrayDesc.Format = texElementDesc.Format;
	texArrayDesc.SampleDesc.Count = 1;
	texArrayDesc.SampleDesc.Quality = 0;
	texArrayDesc.Usage = D3D11_USAGE_DEFAULT;
	texArrayDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	texArrayDesc.CPUAccessFlags = 0;
	texArrayDesc.MiscFlags = 0;

	ID3D11Texture2D* texArray = 0;
	device->CreateTexture2D(&texArrayDesc, 0, &texArray);

	//
	// Copy individual texture elements into texture array.
	//

	// for each texture element...
	for (UINT texElement = 0; texElement < size; ++texElement)
	{
		// for each mipmap level...
		for (UINT mipLevel = 0; mipLevel < texElementDesc.MipLevels; ++mipLevel)
		{
			D3D11_MAPPED_SUBRESOURCE mappedTex2D;
			context->Map(srcTex[texElement], mipLevel, D3D11_MAP_READ, 0, &mappedTex2D);

			context->UpdateSubresource1(texArray, D3D11CalcSubresource(mipLevel, texElement, texElementDesc.MipLevels),
				0, mappedTex2D.pData, mappedTex2D.RowPitch, mappedTex2D.DepthPitch, 0);

			context->Unmap(srcTex[texElement], mipLevel);
		}
	}

	// 
	// Create a resource view to the texture array.
	//

	D3D11_SHADER_RESOURCE_VIEW_DESC viewDesc;
	viewDesc.Format = texArrayDesc.Format;
	viewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
	viewDesc.Texture2DArray.MostDetailedMip = 0;
	viewDesc.Texture2DArray.MipLevels = texArrayDesc.MipLevels;
	viewDesc.Texture2DArray.FirstArraySlice = 0;
	viewDesc.Texture2DArray.ArraySize = size;

	ID3D11ShaderResourceView* texArraySRV = 0;
	device->CreateShaderResourceView(texArray, &viewDesc, &texArraySRV);

	//
	// Cleanup -- we only nead the resource view
	//

	ReleaseCOM(texArray);

	for (UINT i = 0; i < size; ++i)
		ReleaseCOM(srcTex[i]);

	return texArraySRV;
}