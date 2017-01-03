#include "CDLODQuadTree.h"

#include <vector>

CDLODFlatQuadTree::CDLODFlatQuadTree()
{
}


CDLODFlatQuadTree::~CDLODFlatQuadTree()
{
}

HRESULT CDLODFlatQuadTree::Init(ID3D11Device1 * device, int _x, int _z, int _sizeX, int _sizeZ, UINT16 _lods, UINT16 granularity, XMFLOAT3 buffer)
{
	x = _x;
	z = _z;
	sizeX = _sizeX;
	sizeZ = _sizeZ;
	maxLOD = _lods;
	tile_buffer = buffer;

	// lod details
	float viewDistance = _sizeX*4.05f; // sqrt(2)*(10/7)*2
	float morphStart, morphEnd;

	CDLODQuadTreeConsts.gridDim = XMFLOAT3(granularity, granularity / 2.0, 2.0 / granularity);

	for (auto i = 0; i < maxLOD; ++i)
	{
		morphStart = viewDistance*0.85f;
		morphEnd = viewDistance * 1.00f;

		CDLODQuadTreeConsts.LODConsts[i].morphConsts.x = morphEnd / (morphEnd - morphStart);
		CDLODQuadTreeConsts.LODConsts[i].morphConsts.y = 1.0f / (morphEnd - morphStart);
		CDLODQuadTreeConsts.LODConsts[i].distance = viewDistance;
		
		viewDistance /= 2.0;
	}

	D3D11_BUFFER_DESC constantBufferDesc;
	constantBufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	constantBufferDesc.CPUAccessFlags = 0;
	constantBufferDesc.MiscFlags = 0;
	constantBufferDesc.StructureByteStride = 0;
	constantBufferDesc.Usage = D3D11_USAGE_IMMUTABLE;
	constantBufferDesc.ByteWidth = sizeof(CDLODQuadTreeConsts);

	D3D11_SUBRESOURCE_DATA cbData;
	cbData.pSysMem = &CDLODQuadTreeConsts;

	if (FAILED(device->CreateBuffer(&constantBufferDesc, &cbData, &mLODConstsCB))) return false;

	// vertices
	float step = 1.0f / granularity;

	std::vector<XMFLOAT2> vertices((granularity+1) * (granularity+1));

	int n = 0;
	int nx;
	float y = -0.5;
	for (int i = 0; i <= granularity; ++i, y += step)
	{
		nx = 0;
		float x = -0.5f;
		for (int j = 0; j <= granularity; ++j, x += step, ++nx)
		{
			vertices[n++] = XMFLOAT2(x, y);
		}
	}

	mQuadMesh.SetVertices(device, &vertices[0], vertices.size());

	// indices
	std::vector<USHORT> indices(6 * granularity * granularity);

	int nj = 0;
	n = 0;
	for (int j = 0; j < granularity; ++j, ++nj)
	{
		int ni = 0;
		for (int i = 0; i < granularity; ++i, ++ni)
		{
			indices[n++] = ni + (nj + 1) * nx;
			indices[n++] = (ni + 1) + (nj + 1) * nx;
			indices[n++] = (ni + 1) + nj * nx;
			indices[n++] = ni + (nj + 1) * nx;
			indices[n++] = (ni + 1) + nj * nx;
			indices[n++] = ni + nj * nx;
		}
	}

	mQuadMesh.SetIndices(device, &indices[0], indices.size());

	mQuadMesh.SetInstances(device, (QuadType*)nullptr, 900, true);
	
	std::vector<MeshBuffer::Subset> subsets;
	MeshBuffer::Subset sub;
	sub.Id = 0;
	sub.VertexStart = 0;
	sub.VertexCount = vertices.size();
	sub.FaceStart = 0;
	sub.FaceCount = indices.size()/3;
	subsets.push_back(sub);
	mQuadMesh.SetSubsetTable(subsets);

	return S_OK;
}

void CDLODFlatQuadTree::GenerateTree(ID3D11DeviceContext1* mImmediateContext, std::shared_ptr<CameraClass> & Camera)
{
	std::vector<QuadType> instances;
	instances.reserve(160);
 	ParseQuad(Camera, instances, x, z, sizeX, sizeZ, 0);

	if (instances.size())
		mQuadMesh.UpdateInstances(mImmediateContext, &instances[0], instances.size());
	else
		mQuadMesh.UpdateInstances(mImmediateContext, (QuadType*)nullptr, 0);
}

void CDLODFlatQuadTree::Draw(ID3D11DeviceContext1 * mImmediateContext)
{
	mImmediateContext->VSSetConstantBuffers(3, 1, mLODConstsCB.GetAddressOf());
	mQuadMesh.DrawInstanced(mImmediateContext);
}

void CDLODFlatQuadTree::ParseQuad(std::shared_ptr<CameraClass> & Camera, std::vector<QuadType>& instances, int x, int z, int sizeX, int sizeZ, int lod)
{
	if (lod + 1 == maxLOD)
	{
		instances.push_back(QuadType({ x + sizeX / 2.0f, z + sizeZ / 2.0f }, lod, sizeX));
		return;
	}

	BoundingBox quadBox(XMFLOAT3(x + sizeX / 2.0f, 0.0f, z + sizeZ / 2.0f), XMFLOAT3(sizeX / 2.0f + tile_buffer.x, tile_buffer.y, sizeZ / 2.0f + tile_buffer.z));

	ContainmentType contains = Camera->Contains(quadBox);

	if (contains == DISJOINT) return;

	quadBox = BoundingBox(XMFLOAT3(x + sizeX / 2.0f, 0.0f, z + sizeZ / 2.0f), XMFLOAT3(sizeX / 2.0f, 0, sizeZ / 2.0f));

	XMFLOAT3 camPos;
	XMStoreFloat3(&camPos, Camera->GetPosition());
	ContainmentType containsInside = BoundingSphere(camPos, CDLODQuadTreeConsts.LODConsts[lod + 1].distance).Contains(quadBox);

	if (containsInside != DISJOINT)
	{
		ParseQuad(Camera, instances, x, z, sizeX / 2.0f, sizeZ / 2.0f, lod + 1);
		ParseQuad(Camera, instances, x, z + sizeZ / 2.0f, sizeX / 2.0f, sizeZ / 2.0f, lod + 1);
		ParseQuad(Camera, instances, x + sizeX / 2.0f, z, sizeX / 2.0f, sizeZ / 2.0f, lod + 1);
		ParseQuad(Camera, instances, x + sizeX / 2.0f, z + sizeZ / 2.0f, sizeX / 2.0f, sizeZ / 2.0, lod + 1);
	}
	else
	{
		instances.push_back(QuadType({ x + sizeX / 2.0f, z + sizeZ / 2.0f }, lod, sizeX));
	}

	return;
}
