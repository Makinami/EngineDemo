#pragma once

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#include <d3d11_1.h>

#pragma warning(disable:4838)
#include <DirectXMath.h>
#pragma warning(default:4838)

#include <DirectXPackedVector.h>

// Convenience macro for releasing COM objects.
#define ReleaseCOM(x) { if(x){ x->Release(); x = 0; } }
// Convenience macro for deleting objects.
#define SafeDelete(x) { delete x; x = 0; }

#include <string>
#include <fstream>
#include <vector>
#include <algorithm>

using namespace std;
using namespace DirectX;


class TerrainClass {
public:
	struct InitInfo
	{
		// Filename of RAW heightmap data.
		wstring HeightMapFilename;

		//Texture filenames used for texturing the terrain.
		wstring LayarMapFilename0;
		wstring LayarMapFilename1;
		wstring LayarMapFilename2;
		wstring LayarMapFilename3;
		wstring LayarMapFilename4;
		wstring BlandMapFilename;

		// Scale to apply to heights after they have been loaded from the heightmap.
		float HeightScale;

		// Dimensions of the heightmap.
		UINT HeightmapWidth;
		UINT HeightmapHeight; // height

							  // The cell spacing along the x- and z-axes.
		float CellSpacing;
	};

	struct Vertex
	{
		XMFLOAT3 Pos;
		XMFLOAT2 Tex;
		XMFLOAT2 BoundsY;
	};

	float GetWidth() const;
	float GetDepth() const;
	
private:
	void LoadHeighMap();
	void Smooth();
	bool InBounds(int i, int j);
	float Avarage(int i, int j);
	void CalcAllPatchBoundsY();
	void CalcPatchBoundsY(UINT i, UINT j);
	void BuildQuadPatchVB(ID3D11Device1* device);
	void BuildQuadPatchIB(ID3D11Device1* device);
	void BuildHeightmapSRV(ID3D11Device1* device);

	static const int CellsPerPatch = 64;

	ID3D11Buffer* mQuadPatchVB;
	ID3D11Buffer* mQuadPatchIB;

	ID3D11ShaderResourceView* mHeightmapSRV;

	InitInfo mInfo;

	UINT mNumPatchQuadFaces;

	UINT mNumPatchVertRows;
	UINT mNumPatchVertCols;

	vector<XMFLOAT2> mPatchBoundsY;
	vector<float> mHeightmap;
};

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