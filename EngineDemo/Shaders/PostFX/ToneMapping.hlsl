RWTexture2D<float4> frame;

Texture2D<float> luminance;

float3 Uncharted2Tonemap(float3 x)
{
	// http://www.gdcvault.com/play/1012459/Uncharted_2__HDR_Lighting
	// http://filmicgames.com/archives/75 - fixed coefficients
	float A = 0.15; // Shoulder Strength
	float B = 0.50; // Linear Strength
	float C = 0.10; // Linear Angle
	float D = 0.20; // Toe Strength
	float E = 0.02; // Toe Numerator
	float F = 0.30; // Toe Denominator

	return ((x*(A*x + C*B) + D*E) / (x*(A*x + B) + D*F)) - E / F; // E/F = Toe Angle
}

float3 ToneMap(in float3 colour)
{
	float AveLogLum = exp(luminance.Load(int3(0, 0, 10))); // for 1280x720 10th mipmap is enough, but just for the peace of mind
	// TODO: from atributes
	float middleGray = 1.03 - 2 / (2 + log10(AveLogLum + 1)); // 0.18;
	float LumScale = middleGray / AveLogLum;

	float3 ScaledColour = colour * LumScale;

	float whitePoint = 3.0;

	float ExposureBias = 4.0;
	float3 curr = Uncharted2Tonemap(ExposureBias * ScaledColour);
	float3 whiteScale = 1.0f / Uncharted2Tonemap(whitePoint);

	return curr * whiteScale;
}

[numthreads(16, 16, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	frame[DTid.xy] = float4(ToneMap(frame[DTid.xy].rgb), 1.0);
}