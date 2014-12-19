#include "cameraclass.h"

CameraClass::CameraClass()
: mValid(false)
{
	mPosition = XMFLOAT3(0.0f, 0.5f, -1.0f);
	mUp = XMFLOAT3(0.0f, 1.0f, 0.0f);
	mLookAt = XMFLOAT3(0.0f, 0.5f, 0.0f);
}

CameraClass::~CameraClass()
{}

void CameraClass::SetPosition(float x, float y, float z)
{
	mPosition = XMFLOAT3(x, y, z);
	mValid = false;
}

void CameraClass::SetLookAt(float x, float y, float z)
{
	mLookAt = XMFLOAT3(x, y, z);
	mValid = false;
}

void CameraClass::SetUp(float x, float y, float z)
{
	mUp = XMFLOAT3(x, y, z);
	mValid = false;
}

void CameraClass::ChangePosition(float x, float y, float z)
{
	mPosition = XMFLOAT3(mPosition.x + x, mPosition.y + y, mPosition.z + z);
	mValid = false;
}

void CameraClass::ChangeLookAt(float x, float y, float z)
{
	mLookAt = XMFLOAT3(mLookAt.x + x, mLookAt.y + y, mLookAt.z + z);
	mValid = false;
}

void CameraClass::ChangeUp(float x, float y, float z)
{
	mUp = XMFLOAT3(mUp.x + x, mUp.y + y, mUp.z + z);
	mValid = false;
}

XMMATRIX CameraClass::GetViewMatrix()
{
	if (mValid)	return XMLoadFloat4x4(&mViewMatrix);
	else
	{
		XMMATRIX retMatrix = XMMatrixLookAtLH(XMLoadFloat3(&mPosition), XMLoadFloat3(&mLookAt), XMLoadFloat3(&mUp));
		XMStoreFloat4x4(&mViewMatrix, retMatrix);
		mValid = true;
		return retMatrix;
	}
}
