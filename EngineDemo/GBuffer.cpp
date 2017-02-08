#include "GBuffer.h"

#include "Utilities\RenderViewTargetStack.h"
#include "RenderStates.h"

#include "ShaderManager.h"

using namespace std;
using namespace DirectX;

GBufferClass::GBufferClass()
{
}

int GBufferClass::Init(ID3D11Device1 * device, int width, int height)
{
	// SHADERS
	mPixelShader = ShaderManager::Instance()->GetPS("Deferred Shaders::phongLingtingPS");

	D3D11_INPUT_ELEMENT_DESC vertexDesc[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 }
	};

	int numElements = sizeof(vertexDesc) / sizeof(vertexDesc[0]);

	CreateVSAndInputLayout(L"..\\Debug\\Shaders\\Sky\\SkyVS.cso", device, mVertexShader, vertexDesc, numElements, mInputLayout);
	
	// FULL SCREEN QUAD
	vector<XMFLOAT3> patchVertices(4);

	patchVertices[0] = XMFLOAT3(-1.0f, -1.0f, 0.0f);
	patchVertices[1] = XMFLOAT3(-1.0f, 1.0f, 0.0f);
	patchVertices[2] = XMFLOAT3(1.0f, 1.0f, 0.0f);
	patchVertices[3] = XMFLOAT3(1.0f, -1.0f, 0.0f);

	mScreenQuad.SetVertices(device, &patchVertices[0], patchVertices.size());

	vector<USHORT> indices(6);

	indices[0] = 0;
	indices[1] = 1;
	indices[2] = 2;
	indices[3] = 0;
	indices[4] = 2;
	indices[5] = 3;

	mScreenQuad.SetIndices(device, &indices[0], indices.size());

	vector<MeshBuffer::Subset> subsets;
	MeshBuffer::Subset sub;
	sub.Id = 0;
	sub.VertexStart = 0;
	sub.VertexCount = 4;
	sub.FaceStart = 0;
	sub.FaceCount = 2;
	subsets.push_back(sub);
	mScreenQuad.SetSubsetTable(subsets);
	
	// buffer
	D3D11_BUFFER_DESC cbDesc = {};
	cbDesc.Usage = D3D11_USAGE_DYNAMIC;
	cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	cbDesc.MiscFlags = 0;
	cbDesc.StructureByteStride = 0;

	//Precompute(mImmediateContext);

	// render's buffers
	cbDesc.ByteWidth = sizeof(cbPerFrameVSType);

	device->CreateBuffer(&cbDesc, NULL, &cbPerFrameVS);

	cbDesc.ByteWidth = sizeof(cbPerFramePSType);

	device->CreateBuffer(&cbDesc, NULL, &cbPerFramePS);

	OnResize(device, width, height);

	return S_OK;
}

void GBufferClass::Shutdown()
{
	ReleaseCOM(cbPerFrameVS);
	ReleaseCOM(cbPerFramePS);
}

void GBufferClass::SetBufferRTV(ID3D11DeviceContext1 * mImmediateContext) const
{
	RenderTargetStack::Push(mImmediateContext, mGBufferRTV, mDepthStencil->GetDSV(), buffer_count);

	float colour[4] = {};
	mImmediateContext->ClearRenderTargetView(mGBuffer[0]->GetRTV(), colour);
	mImmediateContext->ClearRenderTargetView(mGBuffer[1]->GetRTV(), colour);
	mImmediateContext->ClearDepthStencilView(mDepthStencil->GetDSV(), D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 0.0f, 0);
}

void GBufferClass::UnsetBufferRTV(ID3D11DeviceContext1 * mImmediateContext) const
{
	if (!RenderTargetStack::Pop(mImmediateContext))
	{
		MessageBox(nullptr, L"cannot pop RTV - only one (GBuffer)", nullptr, 0);
	}
}

void GBufferClass::SetBufferSRV(ID3D11DeviceContext1 * mImmediateContext, int first_slot)
{
	mImmediateContext->PSSetShaderResources(first_slot, buffer_count, mGBufferSRV);
	srv_slot = first_slot;
}

void GBufferClass::UnsetBufferSRV(ID3D11DeviceContext1 * mImmediateContext) const
{
	if (srv_slot >= 0)
	{
		ID3D11ShaderResourceView* ppSRVNULL[] = { nullptr, nullptr };
		mImmediateContext->PSSetShaderResources(srv_slot, buffer_count, ppSRVNULL);
	}
}

void GBufferClass::Resolve(ID3D11DeviceContext1 * mImmediateContext, std::shared_ptr<CameraClass> Camera, DirectionalLight & light)
{
	ID3D11ShaderResourceView* ppSRVNULL[4] = { NULL, NULL, NULL, NULL };

	D3D11_MAPPED_SUBRESOURCE mappedResource;
	UINT stride = sizeof(XMFLOAT3);
	UINT offset = 0;

	mImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	mImmediateContext->IASetInputLayout(mInputLayout);

	// VS
	cbPerFrameVSType* dataVS;
	mImmediateContext->Map(cbPerFrameVS, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	dataVS = (cbPerFrameVSType*)mappedResource.pData;

	dataVS->gViewInverse = XMMatrixInverse(nullptr, XMMatrixTranspose(Camera->GetViewRelSun()));
	dataVS->gProjInverse = XMMatrixInverse(nullptr, Camera->GetProjTrans());

	mImmediateContext->Unmap(cbPerFrameVS, 0);

	mImmediateContext->VSSetConstantBuffers(0, 1, &cbPerFrameVS);

	mImmediateContext->VSSetShader(mVertexShader, NULL, 0);

	// PS
	cbPerFramePSType* dataPS;
	mImmediateContext->Map(cbPerFramePS, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	dataPS = (cbPerFramePSType*)mappedResource.pData;

	XMStoreFloat3(&(dataPS->gCameraPos), Camera->GetPositionRelSun());
	dataPS->gSunDir = light.Direction();
	// change light direction to sun direction
	dataPS->gSunDir.x *= -1; dataPS->gSunDir.y *= -1; dataPS->gSunDir.z *= -1;

	mImmediateContext->Unmap(cbPerFramePS, 0);

	mImmediateContext->PSSetConstantBuffers(0, 1, &cbPerFramePS);

	mImmediateContext->PSSetShader(mPixelShader, nullptr, 0);
	this->SetBufferSRV(mImmediateContext);
	mImmediateContext->PSSetShaderResources(2, 1, mDepthStencil->GetAddressOfSRV());

	//mImmediateContext->OMSetDepthStencilState(RenderStates::DepthStencil::WriteNoTestDSS, 0);

	mScreenQuad.Draw(mImmediateContext);

	//mImmediateContext->OMSetDepthStencilState(RenderStates::DepthStencil::DefaultDSS, 0);

	mImmediateContext->PSSetShaderResources(9, 3, ppSRVNULL);

}

HRESULT GBufferClass::OnResize(ID3D11Device1 * device, int renderWidth, int renderHeight)
{
	EXIT_ON_NULL(mGBuffer[0] =
		TextureFactory::CreateTexture(D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE,
			DXGI_FORMAT_R8G8B8A8_UNORM, renderWidth, renderHeight));

	EXIT_ON_NULL(mGBuffer[1] =
		TextureFactory::CreateTexture(D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE,
			DXGI_FORMAT_R16G16_FLOAT, renderWidth, renderHeight));

	mGBufferRTV[0] = mGBuffer[0]->GetRTV();
	mGBufferRTV[1] = mGBuffer[1]->GetRTV();

	mGBufferSRV[0] = mGBuffer[0]->GetSRV();
	mGBufferSRV[1] = mGBuffer[1]->GetSRV();

	// depth stencil
	D3D11_TEXTURE2D_DESC textDesc;

	ZeroMemory(&textDesc, sizeof(textDesc));

	textDesc.Width = renderWidth;
	textDesc.Height = renderHeight;
	textDesc.MipLevels = 1;
	textDesc.ArraySize = 1;
	textDesc.Format = DXGI_FORMAT_R24G8_TYPELESS;
	textDesc.SampleDesc = { 1, 0 };
	textDesc.Usage = D3D11_USAGE_DEFAULT;
	textDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
	textDesc.CPUAccessFlags = 0;
	textDesc.MiscFlags = 0;

	EXIT_ON_NULL(mDepthStencil =
		TextureFactory::CreateTexture(textDesc, { DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_D24_UNORM_S8_UINT, DXGI_FORMAT_R24_UNORM_X8_TYPELESS, DXGI_FORMAT_UNKNOWN }));

	// view port
	mViewPort = {};
	mViewPort.Width = static_cast<float>(renderWidth);
	mViewPort.Height = static_cast<float>(renderHeight);
	mViewPort.MaxDepth = 1.0;

	//MessageBox(nullptr, L"on resize in GBuffer", nullptr, 0);
	return E_NOTIMPL;
}
