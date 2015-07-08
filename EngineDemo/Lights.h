#pragma once

#include <DirectXMath.h>
#include <Windows.h>

using namespace DirectX;

struct DirectionalLightStruct
{
	DirectionalLightStruct() { ZeroMemory(this, sizeof(this)); }

	XMFLOAT4 Ambient;
	XMFLOAT4 Diffuse;
	XMFLOAT4 Specular;
	XMFLOAT3 Direction;
	float Pad; // Pad the last float so we can set an array of lights if we wanted.
};

class DirectionalLight
{
public:
	DirectionalLight();
	DirectionalLight(const XMFLOAT4& _Ambient, const XMFLOAT4& _Diffuse, const XMFLOAT4& _Specular, const XMFLOAT3& _Direction);

	XMFLOAT4 Ambient();
	XMFLOAT4 Ambient(const XMFLOAT4& _Ambient);
	XMFLOAT4 Diffuse();
	XMFLOAT4 Diffuse(const XMFLOAT4& _Diffuse);
	XMFLOAT4 Specular();
	XMFLOAT4 Specular(const XMFLOAT4& _Specular);
	XMFLOAT3 Direction();
	XMFLOAT3 Direction(const XMFLOAT3& _Direction);

	DirectionalLightStruct LightParams();

	void SetLitWorld(XMFLOAT3 _low, XMFLOAT3 _high);

	XMMATRIX GetViewProjTrans();
	XMMATRIX GetMapProjTrans();

private:
	DirectionalLightStruct mLight;

	XMFLOAT4X4 mView;
	XMFLOAT4X4 mProj;
	XMFLOAT4X4 mMap;

	XMFLOAT4X4 mViewProjTrans;
	XMFLOAT4X4 mMapProjTrans;

	bool mValid;
};