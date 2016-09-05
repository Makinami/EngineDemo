#include "HDR.h"

#include <vector>

#include "Utilities\RenderViewTargetStack.h"

using namespace std;
using namespace DirectX;

namespace PostFX
{
	HDR::HDR()
	{
	}

	HDR::HDR(const HDR &)
	{
	}

	HDR::~HDR()
	{
		Shutdown();
	}

	bool HDR::Init(ID3D11Device1 * device, int width, int height)
	{
		D3D11_TEXTURE2D_DESC textDesc;

		ZeroMemory(&textDesc, sizeof(textDesc));

		textDesc.Width = width;
		textDesc.Height = height;
		textDesc.MipLevels = 1;
		textDesc.ArraySize = 1;
		textDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
		textDesc.SampleDesc = { 1, 0 };
		textDesc.Usage = D3D11_USAGE_DEFAULT;
		textDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
		textDesc.CPUAccessFlags = 0;
		textDesc.MiscFlags = 0;

		EXIT_ON_NULL(mHDRText =
					 TextureFactory::CreateTexture(textDesc));

		textDesc.MipLevels = 0;
		textDesc.Format = DXGI_FORMAT_R32_FLOAT;
		textDesc.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;

		EXIT_ON_NULL(mLuminanceText =
					 TextureFactory::CreateTexture(textDesc));

		CreateCSFromFile(L"..\\Debug\\Shaders\\PostFX\\LuminancePass.cso", device, mLuminancePassCS);
		CreateCSFromFile(L"..\\Debug\\Shaders\\PostFX\\ToneMapping.cso", device, mToneMapPassCS);

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

		// pixel
		CreatePSFromFile(L"..\\Debug\\DebugPS.cso", device, mDebugPS);

		// vertex
		D3D11_INPUT_ELEMENT_DESC vertexDesc[] =
		{
			{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
			{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
			{ "TEXCOORD", 1, DXGI_FORMAT_R32G32_FLOAT, 0, 20, D3D11_INPUT_PER_VERTEX_DATA, 0 }
		};

		int numElements = sizeof(vertexDesc) / sizeof(vertexDesc[0]);

		CreateVSAndInputLayout(L"..\\Debug\\DebugVS.cso", device, mDebugVS, vertexDesc, numElements, mDebugIL);

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

	void HDR::Shutdown()
	{
	}

	void HDR::StarRegister(ID3D11DeviceContext1 * mImmediateContext)
	{
		SetRenderTarget(mImmediateContext);
		ClearRenderTarget(mImmediateContext, XMFLOAT4{ 0, 0, 0, 0 });
	}

	void HDR::StopRegister(ID3D11DeviceContext1 * mImmediateContext)
	{
		RenderTargetStack::Pop(mImmediateContext);
	}

	void HDR::Process(ID3D11DeviceContext1 * mImmediateContext)
	{
		ID3D11UnorderedAccessView* uavNULL = nullptr;
		ID3D11ShaderResourceView* srvNULL = nullptr;
		// Luminance Pass
		mImmediateContext->CSSetShaderResources(0, 1, mHDRText->GetAddressOfSRV());
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, mLuminanceText->GetAddressOfUAV(), nullptr);
		mImmediateContext->CSSetShader(mLuminancePassCS, nullptr, 0);

		mImmediateContext->Dispatch(1280 / 16, 720 / 16, 1);

		mImmediateContext->CSSetUnorderedAccessViews(0, 1, &uavNULL, nullptr);
		mImmediateContext->CSSetShaderResources(0, 1, &srvNULL);

		// TODO: check if custum soltion will be faster
		mImmediateContext->GenerateMips(mLuminanceText->GetSRV());

		mImmediateContext->CSSetUnorderedAccessViews(0, 1, mHDRText->GetAddressOfUAV(), nullptr);
		mImmediateContext->CSSetShaderResources(0, 1, mLuminanceText->GetAddressOfSRV());
		mImmediateContext->CSSetShader(mToneMapPassCS, nullptr, 0);

		mImmediateContext->Dispatch(1280 / 16, 720 / 16, 1);

		mImmediateContext->CSSetUnorderedAccessViews(0, 1, &uavNULL, nullptr);
		mImmediateContext->CSSetShaderResources(0, 1, &srvNULL);
	}

	void HDR::Present(ID3D11DeviceContext1 * mImmediateContext)
	{
		UINT stride = sizeof(TerrainClass::Vertex);
		UINT offset = 0;

		ID3D11ShaderResourceView* nulSRV = nullptr;

		mImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		mImmediateContext->IASetVertexBuffers(0, 1, &mScreenQuadVB, &stride, &offset);
		mImmediateContext->IASetIndexBuffer(mScreenQuadIB, DXGI_FORMAT_R16_UINT, 0);

		// Scale and shift quad to lower-right corner.
		XMMATRIX world(
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f);

		//PS
		mImmediateContext->PSSetShaderResources(0, 1, mHDRText->GetAddressOfSRV());
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

		mImmediateContext->PSSetShaderResources(0, 1, &nulSRV);

	}

	void HDR::SetRenderTarget(ID3D11DeviceContext1 * mImmediateContext)
	{
		RenderTargetStack::Push(mImmediateContext, mHDRText->GetAddressOfRTV(), nullptr);
	}

	void HDR::ClearRenderTarget(ID3D11DeviceContext1 * mImmediateContext, DirectX::XMFLOAT4 & _colour)
	{
		float colour[4] = { _colour.x, _colour.y, _colour.z, _colour.w };

		mImmediateContext->ClearRenderTargetView(mHDRText->GetRTV(), colour);
	}
}
