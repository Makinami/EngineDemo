#include "Lights.h"


/*
 * Directional Light definitions
 */
DirectionalLight::DirectionalLight()
	: mValid(0)
{
	mMap = XMFLOAT4X4(
		0.5f, 0.0f, 0.0f, 0.0f,
		0.0f, -0.5f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.5f, 0.5f, 0.0f, 1.0f);
}

DirectionalLight::DirectionalLight(const XMFLOAT4 & _Ambient, const XMFLOAT4 & _Diffuse, const XMFLOAT4 & _Specular, const XMFLOAT3 & _Direction)
	: mValid(0)
{
	mMap = XMFLOAT4X4(
		0.5f, 0.0f, 0.0f, 0.0f,
		0.0f, -0.5f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.5f, 0.5f, 0.0f, 1.0f);

	mLight.Ambient = _Ambient;
	mLight.Diffuse = _Diffuse;
	mLight.Specular = _Specular;
	mLight.Direction = _Direction;
}

XMFLOAT4 DirectionalLight::Ambient()
{
	return mLight.Ambient;
}

XMFLOAT4 DirectionalLight::Ambient(const XMFLOAT4 & _Ambient)
{
	return mLight.Ambient = _Ambient;
}

XMFLOAT4 DirectionalLight::Diffuse()
{
	return mLight.Diffuse;
}

XMFLOAT4 DirectionalLight::Diffuse(const XMFLOAT4 & _Diffuse)
{
	return mLight.Diffuse = _Diffuse;
}

XMFLOAT4 DirectionalLight::Specular()
{
	return mLight.Specular;
}

XMFLOAT4 DirectionalLight::Specular(const XMFLOAT4 & _Specular)
{
	return mLight.Specular = _Specular;
}

XMFLOAT3 DirectionalLight::Direction()
{
	return mLight.Direction;
}

XMFLOAT3 DirectionalLight::Direction(const XMFLOAT3 & _Direction)
{
	return mLight.Direction = _Direction;
}

DirectionalLightStruct DirectionalLight::LightParams()
{
	return mLight;
}

void DirectionalLight::SetLitWorld(XMFLOAT3 _low, XMFLOAT3 _high)
{
	XMVECTOR low = XMLoadFloat3(&_low);
	XMVECTOR high = XMLoadFloat3(&_high);
	XMVECTOR center = (low + high) / 2.0f;

	float radius = XMVectorGetX(XMVector3Length(center - low));

	XMVECTOR lightDir = XMLoadFloat3(&(mLight.Direction));
	XMVECTOR lightPos = -1.0f*(radius + 1.0f)*lightDir;
	XMVECTOR targetPos = XMVectorZero();
	XMVECTOR up = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);

	XMMATRIX V = XMMatrixLookAtLH(lightPos, targetPos, up);

	low = XMVector3TransformCoord(low, V);
	high = XMVector3TransformCoord(high, V);

	float l = min(XMVectorGetX(low), XMVectorGetX(high));
	float b = min(XMVectorGetY(low), XMVectorGetY(high));
	float n = min(XMVectorGetZ(low), XMVectorGetZ(high));
	float r = max(XMVectorGetX(low), XMVectorGetX(high));
	float t = max(XMVectorGetY(low), XMVectorGetY(high));
	float f = max(XMVectorGetZ(low), XMVectorGetZ(high));

	XMMATRIX P = XMMatrixOrthographicOffCenterLH(l, r, b, t, n, f);

	XMStoreFloat4x4(&mView, V);
	XMStoreFloat4x4(&mProj, P);

	XMMATRIX T = XMLoadFloat4x4(&mMap);

	XMStoreFloat4x4(&mViewProjTrans, XMMatrixTranspose(V*P));
	XMStoreFloat4x4(&mMapProjTrans, XMMatrixTranspose(V*P*T));
}

XMMATRIX DirectionalLight::GetViewProjTrans()
{
	return XMLoadFloat4x4(&mViewProjTrans);
}

XMMATRIX DirectionalLight::GetMapProjTrans()
{
	return XMLoadFloat4x4(&mMapProjTrans);
}
