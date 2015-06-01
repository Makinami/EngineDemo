#include "cameraclass.h"

CameraClass::CameraClass()
: mValid(false)
{
	mPosition = XMFLOAT3(0.0f, 10.7f, 0.0f);
	mUp = XMFLOAT3(0.0f, 1.0f, 0.0f);
	mLook = XMFLOAT3(1.0f, 0.0f, 0.0f);
	mRight = XMFLOAT3(0.0f, 0.0f, -1.0f);

	SetLens(XM_PIDIV4, 16.0f / 9.0f, 1.0f, 100.0f);
}

CameraClass::~CameraClass()
{}

void CameraClass::SetPosition(float x, float y, float z)
{
	mPosition = XMFLOAT3(x, y, z);
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

void CameraClass::setRight(float x, float y, float z)
{
	mRight = XMFLOAT3(x, y, z);
	mValid = false;
}

void CameraClass::SetLens(float fovY, float aspect, float zn, float zf)
{
	XMStoreFloat4x4(&mProj, XMMatrixPerspectiveFovLH(fovY, aspect, zn, zf));

	mValid = false;
}

void CameraClass::LookAt(FXMVECTOR pos, FXMVECTOR target, FXMVECTOR worldUp)
{
	XMVECTOR L = XMVector3Normalize(XMVectorSubtract(target, pos));
	XMVECTOR R = XMVector3Normalize(XMVector3Cross(worldUp, L));
	XMVECTOR U = XMVector3Cross(L, R);

	XMStoreFloat3(&mPosition, pos);
	XMStoreFloat3(&mLook, L);
	XMStoreFloat3(&mRight, R);
	XMStoreFloat3(&mUp, U);

	mValid = false;
}

void CameraClass::ChangePosition(float x, float y, float z)
{
	mPosition = XMFLOAT3(mPosition.x + x, mPosition.y + y, mPosition.z + z);
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
	if (mValid)	return XMLoadFloat4x4(&mView);
	else
	{
		UpdateViewMatrix();
		return XMLoadFloat4x4(&mView);
	}
}

XMMATRIX CameraClass::GetProjMatrix()
{
	return XMLoadFloat4x4(&mProj);
}

XMMATRIX CameraClass::GetViewProjMatrix()
{
	if (mValid) return XMLoadFloat4x4(&mViewProj);
	else
	{
		UpdateViewMatrix();
		return XMLoadFloat4x4(&mViewProj);
	}
}

XMMATRIX CameraClass::GetViewProjTransMatrix()
{
	if (mValid) return XMLoadFloat4x4(&mViewProjTrans);
	else
	{
		UpdateViewMatrix();
		return XMLoadFloat4x4(&mViewProjTrans);
	}
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

	mValid = false;
}

void CameraClass::Pitch(float angle)
{
	XMMATRIX R = XMMatrixRotationAxis(XMLoadFloat3(&mRight), angle);

	//XMStoreFloat3(&mUp, XMVector3TransformNormal(XMLoadFloat3(&mUp), R));
	XMStoreFloat3(&mLook, XMVector3TransformNormal(XMLoadFloat3(&mLook), R));

	mValid = false;
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

	XMStoreFloat4x4(&mView, mViewMatrix);
	XMStoreFloat4x4(&mViewProj, mViewProjMatrix);
	XMStoreFloat4x4(&mViewProjTrans, XMMatrixTranspose(mViewProjMatrix));

	mValid = true;
}

