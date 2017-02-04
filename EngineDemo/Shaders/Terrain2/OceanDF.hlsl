Texture2D<float4> heightmap : register(t0);

RWTexture2D<float4> df : register(u0);

SamplerState trilinear
{
	Filter = MIN_MAG_MIP_LINEAR;
	AddressU = Clamp;
	AddressV = Clamp;
};

[numthreads(1, 256, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{

	// Down
	{
		float4 prev, curr = 0.0.xxxx;
		float prevH, currH;
		float2 pos = float2(DTid.y, 0);

		currH = heightmap.SampleLevel(trilinear, (pos + 0.5f) / 256.0f, 0).x;

		if (currH > 0.0f)
		{
			curr.yz = pos - float2(0.0, 1.0);
		}
		else
		{
			curr.yz = pos;
		}

		df[pos] = curr;

		prevH = currH;
		prev = curr;

		for (pos.y = 1; pos.y < 256; ++pos.y)
		{
			currH = heightmap.SampleLevel(trilinear, (pos + 0.5f) / 256.0f, 0).x;

			if (sign(currH) != sign(prevH))
			{
				curr.yz = pos;
			}
			else
			{
				curr.yz = prev.yz;
			}

			df[pos] = curr;

			prev = curr;
			prevH = currH;
		}
	}

	// Right
	{
		float4 prev, curr = 0.0.xxxx;
		float prevH, currH;
		float2 pos = float2(0, DTid.y);

		currH = heightmap.SampleLevel(trilinear, (pos + 0.5f) / 256.0f, 0).x;
		curr.yz = df[pos].yz;

		if (currH > 0.0f)
		{
			curr.yz = pos - float2(1.0, 0.0);
		}
		else
		{
			//curr.yz = pos;
		}

		df[pos] = curr;

		prevH = currH;
		prev = curr;

		for (pos.x = 1; pos.x < 256; ++pos.x)
		{
			currH = heightmap.SampleLevel(trilinear, (pos + 0.5f) / 256.0f, 0).x;
			curr.yz = df[pos].yz;

			if (sign(currH) != sign(prevH))
			{
				curr.yz = pos;
			}
			else
			{
				if (length(pos - curr.yz) > length(pos - prev.yz))
				{
					curr.yz = prev.yz;
				}
			}

			df[pos] = curr;

			prev = curr;
			prevH = currH;
		}
	}

	// Up
	{
		float4 prev, curr = 0.0.xxxx;
		float prevH, currH;
		float2 pos = float2(DTid.y, 255);

		currH = heightmap.SampleLevel(trilinear, (pos + 0.5f) / 256.0f, 0).x;
		curr.yz = df[pos].yz;

		if (currH > 0.0f)
		{
			curr.yz = pos + float2(0.0, 1.0);
		}
		else
		{
			//curr.yz = pos;
		}

		df[pos] = curr;

		prevH = currH;
		prev = curr;

		for (pos.y = 254; pos.y >= 0; --pos.y)
		{
			currH = heightmap.SampleLevel(trilinear, (pos + 0.5f) / 256.0f, 0).x;
			curr.yz = df[pos].yz;

			if (sign(currH) != sign(prevH))
			{
				curr.yz = pos;
			}
			else
			{
				if (length(pos - curr.yz) > length(pos - prev.yz))
				{
					curr.yz = prev.yz;
				}
			}

			df[pos] = curr;

			prev = curr;
			prevH = currH;
		}
	}

	// Left
	{
		float4 prev, curr = 0.0.xxxx;
		float prevH, currH;
		float2 pos = float2(255, DTid.y);

		currH = heightmap.SampleLevel(trilinear, (pos + 0.5f) / 256.0f, 0).x;
		curr.yz = df[pos].yz;

		if (currH > 0.0f)
		{
			curr.yz = pos + float2(1.0, 0.0);
		}
		else
		{
			//curr.yz = pos;
		}

		df[pos] = curr;

		prevH = currH;
		prev = curr;

		for (pos.x = 254; pos.x >= 0; --pos.x)
		{
			currH = heightmap.SampleLevel(trilinear, (pos + 0.5f) / 256.0f, 0).x;
			curr.yz = df[pos].yz;

			if (sign(currH) != sign(prevH))
			{
				curr.yz = pos;
			}
			else
			{
				if (length(pos - curr.yz) > length(pos - prev.yz))
				{
					curr.yz = prev.yz;
				}
			}

			df[pos] = curr;

			prev = curr;
			prevH = currH;
		}
	}
	
	for (int i = 0; i < 256; ++i)
	{
		float4 curr = df[int2(DTid.y, i)];
		float2 gradient = curr.yz - float2(DTid.y, i);
		curr.x = length(gradient);
		curr.yz = 0.0.xx;// normalize(gradient);
		df[int2(DTid.y, i)] = curr;
	}
}