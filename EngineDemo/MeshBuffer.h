#pragma once

#include <d3d11_1.h>

#include <vector>
#include <wrl\client.h>

#include "Utilities\MapResources.h"

class MeshBuffer
{
public:
	struct Subset
	{
		UINT Id;
		UINT VertexStart;
		UINT VertexCount;
		UINT FaceStart;
		UINT FaceCount;

		Subset() : Id(-1), VertexStart(0), VertexCount(0), FaceStart(0), FaceCount(0) {}
	};

public:
	MeshBuffer();
	~MeshBuffer();

	template <typename VertexType>
	HRESULT SetVertices(ID3D11Device1* device, const VertexType* vertices, UINT count);

	HRESULT SetIndices(ID3D11Device1* device, const USHORT* indices, UINT count);

	template <typename InstanceType>
	HRESULT SetInstances(ID3D11Device1* device, const InstanceType* instances, UINT count, bool _mutable = false);

	template <typename InstanceType>
	HRESULT UpdateInstances(ID3D11DeviceContext1* mImmediateContext, const InstanceType* instances, UINT count);

	void SetSubsetTable(std::vector<Subset>& subsetTable);

	void Draw(ID3D11DeviceContext1* mImmediateContext, UINT subsetId = -1);
	void DrawInstanced(ID3D11DeviceContext1* mImmediateContext, UINT subsetId = -1);

private:
	MeshBuffer(const MeshBuffer&) = delete;
	MeshBuffer& operator=(const MeshBuffer&) = delete;

private:
	Microsoft::WRL::ComPtr<ID3D11Buffer> mVertexBuffer;
	Microsoft::WRL::ComPtr<ID3D11Buffer> mIndexBuffer;
	Microsoft::WRL::ComPtr<ID3D11Buffer> mInstanceBuffer;

	DXGI_FORMAT mIndexBufferFormat; // for now 16-bit
	UINT mVertexStride;
	UINT mInstanceStride;
	UINT mInstanceCount;
	UINT offset;

	bool mInstanceBufferMutable;

	std::vector<Subset> mSubsetTable;
};

template<typename VertexType>
inline HRESULT MeshBuffer::SetVertices(ID3D11Device1 * device, const VertexType * vertices, UINT count)
{
	mVertexStride = sizeof(VertexType);

	D3D11_BUFFER_DESC vbd;
	vbd.Usage = D3D11_USAGE_IMMUTABLE;
	vbd.ByteWidth = mVertexStride * count;
	vbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	vbd.CPUAccessFlags = 0;
	vbd.MiscFlags = 0;
	vbd.StructureByteStride = 0;

	D3D11_SUBRESOURCE_DATA vinitData;
	vinitData.pSysMem = vertices;

	return device->CreateBuffer(&vbd, &vinitData, &mVertexBuffer);
}

template<typename InstanceType>
inline HRESULT MeshBuffer::SetInstances(ID3D11Device1 * device, const InstanceType * instances, UINT count, bool _mutable)
{
	mInstanceBufferMutable = _mutable;
	mInstanceCount = count;

	mInstanceStride = sizeof(InstanceType);

	D3D11_BUFFER_DESC ibd;
	ibd.Usage = mInstanceBufferMutable ? D3D11_USAGE_DYNAMIC : D3D11_USAGE_IMMUTABLE;
	ibd.ByteWidth = mInstanceStride * mInstanceCount;
	ibd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	ibd.CPUAccessFlags = mInstanceBufferMutable ? D3D11_CPU_ACCESS_WRITE : 0;
	ibd.MiscFlags = 0;
	ibd.StructureByteStride = 0;

	D3D11_SUBRESOURCE_DATA iinitData;
	iinitData.pSysMem = instances;

	return device->CreateBuffer(&ibd, mInstanceBufferMutable ? nullptr : &iinitData, &mInstanceBuffer);
}

template<typename InstanceType>
inline HRESULT MeshBuffer::UpdateInstances(ID3D11DeviceContext1 * mImmediateContext, const InstanceType * instances, UINT count)
{
	if (mInstanceBufferMutable)
	{
		mInstanceCount = count;
		if (mInstanceCount == 0) return S_OK;

		MapResources(mImmediateContext, mInstanceBuffer.Get(), *instances, mInstanceStride*mInstanceCount);
		return S_OK;
	}
	else return E_ACCESSDENIED;
}
