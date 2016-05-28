#include "cameraclass.h"

CameraClass::CameraClass()
: mValid(false),
pitch(0.0f)
{
	mPosition = XMFLOAT3(0.0f, 10.7f, 0.0f);
	mUp = XMFLOAT3(0.0f, 1.0f, 0.0f);
	mLook = XMFLOAT3(1.0f, 0.0f, 0.0f);
	mRight = XMFLOAT3(0.0f, 0.0f, -1.0f);
	mPositionRelSun = XMFLOAT3(0.0f, 0.0f, 0.0f);

	SetLens(XM_PIDIV4, 16.0f / 9.0f, 1.0f, 100.0f);
}

CameraClass::~CameraClass()
{}

void CameraClass::SetPosition(float x, float y, float z)
{
	mPosition = XMFLOAT3(x, y, z);
	mPositionRelSun = XMFLOAT3(0.0f, 6360 + y/1000, 0.0f);
	mValid = false;
}

void CameraClass::SetLook(float x, float y, float z)
{
	mLook = XMFLOAT3(x, y, z);
	mValid = false;
}

void CameraClass::SetUp(float x, float y, float z)
{
	mUp = XMFLOAT3(x, y, z);
	mValid = false;
}

void CameraClass::SetRight(float x, float y, float z)
{
	mRight = XMFLOAT3(x, y, z);
	mValid = false;
}

void CameraClass::SetLens(float fovY, float aspect, float zn, float zf)
{
	XMMATRIX proj = XMMatrixPerspectiveFovLH(fovY, aspect, zn, zf);

	XMStoreFloat4x4(&mProj, proj);
	XMStoreFloat4x4(&mProjTrans, XMMatrixTranspose(proj));

	mValid = false;
}

void CameraClass::LookAt(FXMVECTOR pos, FXMVECTOR target, FXMVECTOR worldUp)
{
	XMVECTOR L = XMVector3Normalize(XMVectorSubtract(target, pos));
	XMVECTOR R = XMVector3Normalize(XMVector3Cross(worldUp, L));
	XMVECTOR U = XMVector3Cross(L, R);

	XMStoreFloat3(&mPosition, pos);
	mPositionRelSun.y = mPosition.y;

	XMStoreFloat3(&mLook, L);
	XMStoreFloat3(&mRight, R);
	XMStoreFloat3(&mUp, U);

	mValid = false;
}

void CameraClass::ChangePosition(float x, float y, float z)
{
	mPosition = XMFLOAT3(mPosition.x + x, mPosition.y + y, mPosition.z + z);
	mPositionRelSun.y = mPosition.y / 1000 + 6360.0f;
	mValid = false;
}

void CameraClass::ChangeLookAt(float x, float y, float z)
{
	mLook = XMFLOAT3(mLook.x + x, mLook.y + y, mLook.z + z);
	mValid = false;
}

void CameraClass::ChangeUp(float x, float y, float z)
{
	mUp = XMFLOAT3(mUp.x + x, mUp.y + y, mUp.z + z);
	mValid = false;
}

XMVECTOR CameraClass::GetPosition() const
{
	return XMLoadFloat3(&mPosition);
}

XMVECTOR CameraClass::GetAhead() const
{
	XMVECTOR u = XMLoadFloat3(&mUp);
	XMVECTOR r = XMLoadFloat3(&mRight);
	return XMVector3Cross(r, u);
}

XMVECTOR CameraClass::GetRight() const
{
	return XMLoadFloat3(&mRight);
}

XMMATRIX CameraClass::GetViewMatrix()
{
	if (!mValid) UpdateViewMatrix();
	return XMLoadFloat4x4(&mView);
}

XMMATRIX CameraClass::GetProjMatrix()
{
	return XMLoadFloat4x4(&mProj);
}

XMMATRIX CameraClass::GetViewProjMatrix()
{
	if (!mValid) UpdateViewMatrix();
	return XMLoadFloat4x4(&mViewProj);
}

XMMATRIX CameraClass::GetViewProjTransMatrix()
{
	if (mValid) UpdateViewMatrix();
	return XMLoadFloat4x4(&mViewProjTrans);
}

XMMATRIX CameraClass::GetViewRelSun()
{
	if (!mValid) UpdateViewMatrix();
	return XMLoadFloat4x4(&mViewRelSun);
}

XMMATRIX CameraClass::GetViewRelSunTrans()
{
	if (!mValid) UpdateViewMatrix();
	return XMLoadFloat4x4(&mViewRelSunTrans);
}

XMMATRIX CameraClass::GetProjTrans()
{
	return XMLoadFloat4x4(&mProjTrans);
}

XMVECTOR CameraClass::GetPositionRelSun() const
{
	return XMLoadFloat3(&mPositionRelSun);
}

void CameraClass::Walk(XMFLOAT3 deltaX)
{
	XMVECTOR w = XMVectorReplicate(deltaX.y);
	XMVECTOR d = XMVectorReplicate(deltaX.x);
	XMVECTOR l = XMLoadFloat3(&mLook);
	XMVECTOR r = XMLoadFloat3(&mRight);
	XMVECTOR p = XMLoadFloat3(&mPosition);
	
	p = XMVectorMultiplyAdd(w, l, p);
	XMStoreFloat3(&mPosition, XMVectorMultiplyAdd(d, r, p));
	mPositionRelSun.y = mPosition.y/1000 + 6360.0f;

	mValid = false;
}

void CameraClass::Pitch(float angle)
{
	XMMATRIX R = XMMatrixRotationAxis(XMLoadFloat3(&mRight), angle);

	//XMStoreFloat3(&mUp, XMVector3TransformNormal(XMLoadFloat3(&mUp), R));
	XMStoreFloat3(&mLook, XMVector3TransformNormal(XMLoadFloat3(&mLook), R));

	mValid = false;

	// temp?
	pitch += angle;
}

float CameraClass::GetHorizon()
{
	float R = 6360000.0f;
	float z = mPosition.y;
	float a = -sqrt((2 * R*z + z*z) / (R*R));
	float angle = atan(a);

	XMMATRIX Rot = XMMatrixRotationAxis(XMLoadFloat3(&mRight), XM_PIDIV2-angle);
	XMVECTOR look = XMVector3TransformNormal(XMLoadFloat3(&mUp), Rot);
	look = XMLoadFloat3(&mPosition) + look;
	look = XMVector3Project(look, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, GetProjMatrix(), GetViewMatrix(), XMMatrixIdentity());

	return 1.0f - XMVectorGetY(look);
}

void CameraClass::RotateY(float angle)
{
	XMMATRIX R = XMMatrixRotationY(angle);

	XMStoreFloat3(&mRight, XMVector3TransformNormal(XMLoadFloat3(&mRight), R));
	XMStoreFloat3(&mLook, XMVector3TransformNormal(XMLoadFloat3(&mLook), R));
	XMStoreFloat3(&mUp, XMVector3TransformNormal(XMLoadFloat3(&mUp), R));

	mValid = false;
}

inline void CameraClass::UpdateViewMatrix()
{
	XMMATRIX mViewMatrix = XMMatrixLookToLH(XMLoadFloat3(&mPosition), XMLoadFloat3(&mLook), XMLoadFloat3(&mUp));
	XMMATRIX mViewProjMatrix = mViewMatrix*GetProjMatrix();
	XMMATRIX mViewRelSunMatrix = XMMatrixLookToLH(XMLoadFloat3(&mPositionRelSun), XMLoadFloat3(&mLook), XMLoadFloat3(&mUp));

	XMStoreFloat4x4(&mView, mViewMatrix);
	XMStoreFloat4x4(&mViewProj, mViewProjMatrix);
	XMStoreFloat4x4(&mViewProjTrans, XMMatrixTranspose(mViewProjMatrix));
	XMStoreFloat4x4(&mViewRelSun, mViewRelSunMatrix);
	XMStoreFloat4x4(&mViewRelSunTrans, XMMatrixTranspose(mViewRelSunMatrix));

	mValid = true;
}

