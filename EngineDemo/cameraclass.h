#pragma once

#include <d3d11_1.h>

#pragma warning(disable:4838)
#include <DirectXMath.h>
#pragma warning(default:4838)

using namespace DirectX;

class CameraClass
{
	public:
		CameraClass();
		~CameraClass();

		void SetPosition(float x, float y, float z);
		void SetLookAt(float x, float y, float z);
		void SetUp(float x, float y, float z);

		void ChangePosition(float x, float y, float z);
		void ChangeLookAt(float x, float y, float z);
		void ChangeUp(float x, float y, float z);

		XMMATRIX GetViewMatrix();

	private:
		XMFLOAT3 mPosition, mUp, mLookAt;
		XMFLOAT4X4 mViewMatrix;
		bool mValid;
};