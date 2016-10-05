#include "CDLODQuadTree.h"

#include <vector>

CDLODFlatQuadTree::CDLODFlatQuadTree()
{
}


CDLODFlatQuadTree::~CDLODFlatQuadTree()
{
}

HRESULT CDLODFlatQuadTree::Init(ID3D11Device1 * device, int _x, int _z, int _sizeX, int _sizeZ, UINT16 _lods, UINT16 granularity)
{
	x = _x;
	z = _z;
	sizeX = _sizeX;
	sizeZ = _sizeZ;
	maxLOD = _lods;

	// lod details
	float viewDistance = 40960.0f * 5.0;
	float currentSize = 40960.0f * 2.0;
	float morphStart, morphEnd;
	LODConstsStruct currentLOD;

	for (auto i = 0; i < maxLOD; ++i)
	{
		morphStart = viewDistance*0.85 * 2.0;
		morphEnd = viewDistance * 1.05* 2.0;

		currentLOD.size = currentSize;
		currentLOD.morphConsts.x = morphEnd / (morphEnd - morphStart);
		currentLOD.morphConsts.y = 1.0 / (morphEnd - morphStart);
		currentLOD.distance = viewDistance;

		mLODConsts.push_back(currentLOD);

		viewDistance /= 2.0;
		currentSize /= 2.0;
	}

	D3D11_BUFFER_DESC constantBufferDesc;
	constantBufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	constantBufferDesc.CPUAccessFlags = 0;
	constantBufferDesc.MiscFlags = 0;
	constantBufferDesc.StructureByteStride = 0;
	constantBufferDesc.Usage = D3D11_USAGE_IMMUTABLE;
	constantBufferDesc.ByteWidth = mLODConsts.size() * sizeof(LODConstsStruct);

	D3D11_SUBRESOURCE_DATA cbData;
	cbData.pSysMem = &mLODConsts[0];

	if (FAILED(device->CreateBuffer(&constantBufferDesc, &cbData, &mLODConstsCB))) return false;

	// vertices
	float step = 1.0 / granularity;

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
			indices[n++] = (ni + 1) + nj * nx;
			indices[n++] = ni + (nj + 1) * nx;
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
	if (lod == maxLOD - 1)
	{
		instances.push_back(QuadType({ x + sizeX / 2.0f, z + sizeZ / 2.0f }, lod));
		return;
	}

	BoundingBox quadBox(XMFLOAT3(x+sizeX/2.0, 0.0, z+sizeZ/2.0), XMFLOAT3(sizeX/2.0, 15.0, sizeZ/2.0));

	ContainmentType contains = Camera->Contains(quadBox);

	if (contains == DISJOINT) return;

	XMFLOAT3 camPos;
	XMStoreFloat3(&camPos, Camera->GetPosition());
	BoundingSphere lodSphere(camPos, mLODConsts[lod].distance);

	contains = lodSphere.Contains(quadBox);
	if (contains == DISJOINT)
	{
		instances.push_back(QuadType( { x + sizeX / 2.0f, z + sizeZ / 2.0f }, lod));
	}
	else
	{
		ParseQuad(Camera, instances, x, z, sizeX / 2.0, sizeZ / 2.0, lod + 1);
		ParseQuad(Camera, instances, x, z + sizeZ / 2.0, sizeX / 2.0, sizeZ / 2.0, lod + 1);
		ParseQuad(Camera, instances, x + sizeX / 2.0, z, sizeX / 2.0, sizeZ / 2.0, lod + 1);
		ParseQuad(Camera, instances, x + sizeX / 2.0, z + sizeZ / 2.0, sizeX / 2.0, sizeZ / 2.0, lod + 1);
	}

	return;
}
