#include "playerclass.h"

PlayerClass::PlayerClass()
{
	mPosition = XMFLOAT3(0.0f, 0.0f, 0.0f);
}

PlayerClass::~PlayerClass()
{
}

void PlayerClass::SetTerrain(std::shared_ptr<TerrainClass> iMap)
{
	Terrain = iMap;
}

void PlayerClass::Init()
{
	mPosition.y = Terrain->GetHeight(mPosition.x, mPosition.z);
	Camera->SetPosition(mPosition.x, mPosition.y + 1.7f, mPosition.z);
}

void PlayerClass::SetCamera(std::shared_ptr<CameraClass> iCamera)
{
	Camera = iCamera;
}

void PlayerClass::SetInput(std::shared_ptr<InputClass> iInput)
{
	Input = iInput;
}

void PlayerClass::React(float dt)
{
	if (Input->IsDrag())
	{
		POINT mDrag = Input->GetDragDelta();
		float dx = XMConvertToRadians(0.25f*static_cast<float>(mDrag.x));
		float dy = XMConvertToRadians(0.25f*static_cast<float>(mDrag.y));

		Camera->RotateY(dx);
		Camera->Pitch(dy);
	}

	XMFLOAT3 deltaX = { 0.0f, 0.0f, 0.0f };

	if (Input->IsKeyPressed('W')) deltaX.z += 1.0f;
	if (Input->IsKeyPressed('S')) deltaX.z -= 1.0f;
	if (Input->IsKeyPressed('A')) deltaX.x -= 1.0f;
	if (Input->IsKeyPressed('D')) deltaX.x += 1.0f;

	if (deltaX.x || deltaX.y || deltaX.z)
	{
		float speed = (Input->IsKeyPressed(VK_LSHIFT) ? RunV : WalkV);

		XMVECTOR Ahead = Camera->GetAhead();
		XMVECTOR Right = Camera->GetRight();
		XMVECTOR Position = XMLoadFloat3(&mPosition);
		XMStoreFloat3(&deltaX, speed*dt*XMVector3Normalize(XMLoadFloat3(&deltaX)));
		XMVECTOR w = XMVectorReplicate(deltaX.z);
		XMVECTOR d = XMVectorReplicate(deltaX.x);

		XMVECTOR PosTemp = XMVectorMultiplyAdd(w, Ahead, Position);
		PosTemp = XMVectorMultiplyAdd(d, Right, PosTemp);
		XMFLOAT3 fPosTemp;
		XMStoreFloat3(&fPosTemp, PosTemp);
		deltaX.x = fPosTemp.x - mPosition.x;
		deltaX.z = fPosTemp.z - mPosition.z;
		deltaX.y = Terrain->GetHeight(fPosTemp.x, fPosTemp.z) - Terrain->GetHeight(mPosition.x, mPosition.z);
		XMStoreFloat3(&deltaX, speed*dt*XMVector3Normalize(XMLoadFloat3(&deltaX)));
		mPosition.x += deltaX.x;
		mPosition.z += deltaX.z;
		mPosition.y = Terrain->GetHeight(mPosition.x, mPosition.z);
		Camera->SetPosition(mPosition.x, mPosition.y + 1.7f, mPosition.z);
	}
}
