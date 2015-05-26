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

#include "loggerclass.h"

using namespace std;
using namespace DirectX;


class TerrainClass : public HasLogger
{
public:
	struct InitInfo
	{
		// Filename of RAW heightmap data.
		wstring HeightMapFilename;

		//Texture filenames used for texturing the terrain.
		wstring LayerMapFilename0;
		wstring LayerMapFilename1;
		wstring LayerMapFilename2;
		wstring LayerMapFilename3;
		wstring LayerMapFilename4;
		wstring BlendMapFilename;

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

public:
	TerrainClass();
	~TerrainClass();

	float GetWidth() const;
	float GetDepth() const;

	XMMATRIX GetWorld() const;
	void SetWorld(CXMMATRIX M);

	bool Init(ID3D11Device1* device, ID3D11DeviceContext1* dc, const InitInfo& initInfo);

	void Draw(ID3D11DeviceContext1* dc, CXMMATRIX cam, XMVECTOR pos);

private:
	struct MatrixBufferType
	{
		XMMATRIX gWorldProj;
	};

	struct cbPerFrameHSType
	{
		XMFLOAT4 gWorldFrustumPlanes[6];

		XMFLOAT3 gEyePosW;

		// When distance is minimum, the tessellation is maximum.
		// When distance is maximum, the tessellation is minimum.
		float gMinDist;
		float gMaxDist;

		// Exponents for power of 2 tessellation.  The tessellation
		// range is [2^(gMinTess), 2^(gMaxTess)].  Since the maximum
		// tessellation is 64, this means gMaxTess can be at most 6
		// since 2^6 = 64.
		float gMinTess;
		float gMaxTess;

		float padding[1];
	};

	struct cbPerFramePSType
	{
		XMMATRIX gViewProj;

		//DirectionalLight gDirLights[3];
		XMFLOAT3 gEyePosW;

		float gTexelCellSpaceU;
		float gTexelCellSpaceV;
		float gWorldCellSpace;

		float padding[2];
	};
private:
	void LoadHeighMap();
	void Smooth();
	bool InBounds(int i, int j);
	float Avarage(int i, int j);
	void CalcAllPatchBoundsY();
	void CalcPatchBoundsY(UINT i, UINT j);
	void BuildQuadPatchVB(ID3D11Device1* device);
	bool BuildQuadPatchIB(ID3D11Device1* device);
	void BuildHeightmapSRV(ID3D11Device1* device);
	bool CreateInputLayoutAndShaders(ID3D11Device1* device);

	static const int CellsPerPatch = 64;

	ID3D11Buffer* mQuadPatchVB;
	ID3D11Buffer* mQuadPatchIB;

	ID3D11Buffer* cbPerFrameHS;
	ID3D11Buffer* MatrixBuffer;
	ID3D11Buffer* cbPerFramePS;

	ID3D11ShaderResourceView* mLayerMapArraySRV;
	ID3D11ShaderResourceView* mBlendMapSRV;
	ID3D11ShaderResourceView* mHeightmapSRV;

	ID3D11InputLayout* mInputLayout;
	ID3D11VertexShader* mVertexShader;
	ID3D11HullShader* mHullShader;
	ID3D11DomainShader* mDomainShader;
	ID3D11PixelShader* mPixelShader;

	ID3D11RasterizerState* mRastState;
	ID3D11SamplerState** mSamplerStates;

	InitInfo mInfo;

	UINT mNumPatchVertices;
	UINT mNumPatchQuadFaces;

	UINT mNumPatchVertRows;
	UINT mNumPatchVertCols;

	XMFLOAT4X4 mWorld;

	vector<XMFLOAT2> mPatchBoundsY;
	vector<float> mHeightmap;
};


//temp
void ExtractFrustrumPlanes(XMFLOAT4 planes[6], CXMMATRIX M);

ID3D11ShaderResourceView* CreateTexture2DArraySRV(
	ID3D11Device1* device, ID3D11DeviceContext1* context,
	std::vector<std::wstring>& filenames);