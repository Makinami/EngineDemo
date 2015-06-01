#pragma once

#if defined(DEBUG) || defined(_DEBUG)
#define CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif

#include <d3d11_1.h>

#pragma warning(disable:4838)
#include <DirectXMath.h>
#pragma warning(default:4838)

#include <DirectXPackedVector.h>

#define ReleaseCOM(x) { if(x){ x->Release(); x = nullptr; } }
#define SafeDelete(x) { delete x; x = nullptr; }

#include <vector>

#include "loggerclass.h"
#include "cameraclass.h"

using namespace std;
using namespace DirectX;

class WaterClass : public HasLogger
{
public:
	struct Vertex
	{
		XMFLOAT3 Pos;
	};

public:
	WaterClass();
	~WaterClass();

	bool Init(ID3D11Device1* device, ID3D11DeviceContext1* dc);

	void Draw(ID3D11DeviceContext1* mImmediateContext, std::shared_ptr<CameraClass> Camera);

private:
	struct MatrixBufferType
	{
		XMMATRIX gWorld;
	};

private:
	void BuildQuadPatchVB(ID3D11Device1* device);
	bool BuildQuadPatchIB(ID3D11Device1* device);
	bool CreateInputLayoutAndShaders(ID3D11Device1* device);

	ID3D11Buffer* mQuadPatchVB;
	ID3D11Buffer* mQuadPatchIB;

	ID3D11Buffer* MatrixBuffer;

	ID3D11InputLayout* mInputLayout;
	ID3D11VertexShader* mVertexShader;
	ID3D11PixelShader* mPixelShader;

	XMFLOAT4X4 mWorld;
};