#pragma once

#define CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

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
		void SetLook(float x, float y, float z);
		void SetUp(float x, float y, float z);
		void SetRight(float x, float y, float z);

		void SetLens(float fovY, float aspect, float zn, float zf);

		void LookAt(FXMVECTOR pos, FXMVECTOR target, FXMVECTOR worldUp);

		void ChangePosition(float x, float y, float z);
		void ChangeLookAt(float x, float y, float z);
		void ChangeUp(float x, float y, float z);

		XMVECTOR GetPosition() const;
		XMVECTOR GetAhead() const;
		XMVECTOR GetRight() const;

		XMMATRIX GetViewMatrix();
		XMMATRIX GetProjMatrix();
		XMMATRIX GetViewProjMatrix();
		XMMATRIX GetViewProjTransMatrix();

		XMMATRIX GetViewRelSun();
		XMMATRIX GetViewRelSunTrans();
		XMMATRIX GetProjTrans();
		XMVECTOR GetPositionRelSun() const;

		void Walk(XMFLOAT3 d);

		void Pitch(float angle);
		void RotateY(float angle);

	private:

		inline void UpdateViewMatrix();

		XMFLOAT3 mPosition, mRight, mUp, mLook;

		XMFLOAT3 mPositionRelSun;
		
		XMFLOAT4X4 mProj;
		XMFLOAT4X4 mView;
		XMFLOAT4X4 mViewProj;
		XMFLOAT4X4 mViewProjTrans;

		XMFLOAT4X4 mViewRelSun;
		XMFLOAT4X4 mViewRelSunTrans;
		XMFLOAT4X4 mProjTrans;

		bool mValid;
};