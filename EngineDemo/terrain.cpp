#include "terrain.h"

float TerrainClass::GetWidth() const
{
	return (mInfo.HeightmapWidth - 1)*mInfo.CellSpacing;
}

float TerrainClass::GetDepth() const
{
	return (mInfo.HeightmapHeight - 1)*mInfo.CellSpacing;
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
	for (const auto &i : in)
	{
		mHeightmap[i] = (i / 255.0f)*mInfo.HeightScale;
	}
}

void TerrainClass::Smooth()
{
	vector<float> dest(mHeightmap.size());

	for (int i = 0; i < mInfo.HeightmapHeight; ++i)
		for (int j = 0; j < mInfo.HeightmapWidth; ++j)
			dest[i*mInfo.HeightmapWidth + j] = Avarage(i, j);

	mHeightmap = dest;
}

bool TerrainClass::InBounds(int i, int j)
{
	// True if ij are valid indices; false otherwise.
	return
		i >= 0 && (int)mInfo.HeightmapHeight &&
		j >= 0 && (int)mInfo.HeightmapWidth;
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

void TerrainClass::BuildQuadPatchIB(ID3D11Device1* device)
{
	vector<USHORT> indices(mNumPatchQuadFaces * 4);

	int k = 0;
	for (UINT i = 0; i < mNumPatchVertRows - 1; ++i)
	{
		for (UINT j = 0; j < mNumPatchVertCols - 1; ++i)
		{
			indices[k] = i*mNumPatchVertCols + j;
			indices[k + 1] = i*mNumPatchVertCols + j + 1;
			indices[k + 2] = (i + 1)*mNumPatchVertCols + j;
			indices[k + 3] = (i + 1)*mNumPatchVertCols + j;

			k += 4;
		}
	}

	D3D11_BUFFER_DESC ibd;
	ibd.Usage = D3D11_USAGE_IMMUTABLE;
	ibd.ByteWidth = sizeof(USHORT)*indices.size();
	ibd.CPUAccessFlags = 0;
	ibd.MiscFlags = 0;
	ibd.StructureByteStride = 0;

	D3D11_SUBRESOURCE_DATA iinitData;
	iinitData.pSysMem = &indices[0];
	device->CreateBuffer(&ibd, &iinitData, &mQuadPatchIB);
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
	texDesc.SampleDesc.Quality = 1;
	texDesc.Usage = D3D11_USAGE_DEFAULT;
	texDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	texDesc.CPUAccessFlags = 0;
	texDesc.MiscFlags = 0;

	vector<PackedVector::HALF> hmap(mHeightmap.size());
	transform(mHeightmap.begin(), mHeightmap.end(), hmap.begin(), PackedVector::XMConvertFloatToHalf);

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