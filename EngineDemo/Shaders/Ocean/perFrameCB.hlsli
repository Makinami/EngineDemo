cbuffer perFrameCB : register(b1)
{
	matrix screenToCamMatrix;
	matrix camToWorldMatrix;
	matrix worldToScreenMatrix;
	float3 camPos;
	float time;
	float3 sunDir;
	float dt;
	float coneAngle;
	float2 gridSize;
	float lambdaJ;
	float2 sigma2;
	float lambdaV;
	float scale;
	float3 camLookAt;
	float mipmap;
	float4 gProj;
	float2 gbWind;
	float depthClamp;
	float colourAlpha;
	float4 seaColour;
	float4 seaColourSSS;
	int SkyFlag;
	int SeaFlag;
	int SunFlag;
	int varianceSlice;
	float argHDependency;
	float3 rgbExtinction;
}