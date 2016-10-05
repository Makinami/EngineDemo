#pragma once

#include <vector>
#include <wrl\client.h>
#include <memory>

#include "cameraclass.h"
#include "MeshBuffer.h"

class CDLODFlatQuadTree
{
public:
	CDLODFlatQuadTree();
	~CDLODFlatQuadTree();

	HRESULT Init(ID3D11Device1* device, int _x, int _z, int _sizeX, int _sizeZ, UINT16 _lods, UINT16 granularity);

	void GenerateTree(ID3D11DeviceContext1* mImmediateContext, std::shared_ptr<CameraClass> & Camera);

	void Draw(ID3D11DeviceContext1* mImmediateContext);

private:
	struct QuadType
	{
		XMFLOAT2 offset;
		UINT LODid;
		float pad;
		QuadType(XMFLOAT2 _offset, UINT _LODid) : offset(_offset), LODid(_LODid) {};
	};

	struct LODConstsStruct
	{
		float size;
		XMFLOAT2 morphConsts;
		float distance;
	};

	std::vector<LODConstsStruct> mLODConsts;

	void ParseQuad(std::shared_ptr<CameraClass> & Camera, std::vector<QuadType>& instances, int x, int z, int sizeX, int sizeZ, int lod);

	MeshBuffer mQuadMesh;

	Microsoft::WRL::ComPtr<ID3D11Buffer> mLODConstsCB;

	int x;
	int z;
	int sizeX;
	int sizeZ;
	UINT maxLOD;
};

