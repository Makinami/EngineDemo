#include "MeshBuffer.h"

#define ReleaseCOM(x) { if(x){ x->Release(); x = nullptr; } }
#define SafeDelete(x) { delete x; x = nullptr; }

#define EXIT_ON_FAILURE(fnc)  \
	{ \
		HRESULT result; \
		if (FAILED(result = fnc)) { \
			return result; \
		} \
	}

MeshBuffer::MeshBuffer()
	: mVertexBuffer(nullptr), mIndexBuffer(nullptr), 
	mIndexBufferFormat(DXGI_FORMAT_R16_UINT),
	mVertexStride(0), offset(0)
{
}

MeshBuffer::~MeshBuffer()
{
}

HRESULT MeshBuffer::SetIndices(ID3D11Device1 * device, const USHORT * indices, UINT count)
{
	D3D11_BUFFER_DESC ibd;
	ibd.Usage = D3D11_USAGE_IMMUTABLE;
	ibd.ByteWidth = sizeof(USHORT) * count;
	ibd.BindFlags = D3D11_BIND_INDEX_BUFFER;
	ibd.CPUAccessFlags = 0;
	ibd.MiscFlags = 0;
	ibd.StructureByteStride = 0;

	D3D11_SUBRESOURCE_DATA iinitData;
	iinitData.pSysMem = indices;

	return device->CreateBuffer(&ibd, &iinitData, &mIndexBuffer); // HRESULT
}

void MeshBuffer::SetSubsetTable(std::vector<Subset>& subsetTable)
{
	mSubsetTable = subsetTable;
}

void MeshBuffer::Draw(ID3D11DeviceContext1 * mImmediateContext, UINT subsetId)
{
	// TODO: multiple buffer, instancing
	mImmediateContext->IASetVertexBuffers(0, 1, mVertexBuffer.GetAddressOf(), &mVertexStride, &offset);
	mImmediateContext->IASetIndexBuffer(mIndexBuffer.Get(), mIndexBufferFormat, 0);

	if (subsetId != -1)
		mImmediateContext->DrawIndexed(mSubsetTable[subsetId].FaceCount * 3,
									   mSubsetTable[subsetId].FaceStart * 3,
									   0);
	else
		for (auto& subset : mSubsetTable)
			mImmediateContext->DrawIndexed(subset.FaceCount * 3,
										   subset.FaceStart * 3,
										   0);
}

void MeshBuffer::DrawInstanced(ID3D11DeviceContext1 * mImmediateContext, UINT subsetId)
{
	mImmediateContext->IASetVertexBuffers(0, 1, mVertexBuffer.GetAddressOf(), &mVertexStride, &offset);
	mImmediateContext->IASetVertexBuffers(1, 1, mInstanceBuffer.GetAddressOf(), &mInstanceStride, &offset);
	mImmediateContext->IASetIndexBuffer(mIndexBuffer.Get(), mIndexBufferFormat, 0);

	if (subsetId != -1)
		mImmediateContext->DrawIndexedInstanced(mSubsetTable[subsetId].FaceCount * 3, 
												mInstanceCount,
												mSubsetTable[subsetId].FaceStart * 3, 
												0, 0);
	else
		for (auto& subset : mSubsetTable)
			mImmediateContext->DrawIndexedInstanced(subset.FaceCount * 3, 
													mInstanceCount,
													subset.FaceStart * 3, 
													0, 0);
}
