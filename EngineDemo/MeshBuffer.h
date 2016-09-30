#pragma once

#include <d3d11_1.h>

#include <vector>
#include <wrl\client.h>

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

	void SetSubsetTable(std::vector<Subset>& subsetTable);

	void Draw(ID3D11DeviceContext1* mImmediateContext, UINT subsetId = -1);

private:
	MeshBuffer(const MeshBuffer&) = delete;
	MeshBuffer& operator=(const MeshBuffer&) = delete;

private:
	Microsoft::WRL::ComPtr<ID3D11Buffer> mVertexBuffer;
	Microsoft::WRL::ComPtr<ID3D11Buffer> mIndexBuffer;

	DXGI_FORMAT mIndexBufferFormat; // for now 16-bit
	UINT mVertexStride;
	UINT offset;

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
