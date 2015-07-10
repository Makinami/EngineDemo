#include "water.h"

WaterClass::WaterClass() :
	mQuadPatchVB(nullptr),
	mQuadPatchIB(nullptr),
	MatrixBuffer(nullptr),
	mInputLayout(nullptr),
	mVertexShader(nullptr),
	mPixelShader(nullptr)
{
	XMStoreFloat4x4(&mWorld, XMMatrixIdentity());
}

WaterClass::~WaterClass()
{
	ReleaseCOM(mQuadPatchIB);
	ReleaseCOM(mQuadPatchVB);

	ReleaseCOM(MatrixBuffer);

	ReleaseCOM(mInputLayout);
	ReleaseCOM(mVertexShader);
	ReleaseCOM(mPixelShader);

	ReleaseCOM(mRastStateFrame);
}

bool WaterClass::Init(ID3D11Device1 * device, ID3D11DeviceContext1 * dc)
{
	if (!CreateInputLayoutAndShaders(device)) return false;

	// VS buffers
	D3D11_BUFFER_DESC matrixBufferDesc;
	matrixBufferDesc.Usage = D3D11_USAGE_DYNAMIC;
	matrixBufferDesc.ByteWidth = sizeof(MatrixBufferType);
	matrixBufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	matrixBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	matrixBufferDesc.MiscFlags = 0;
	matrixBufferDesc.StructureByteStride = 0;

	if (FAILED(device->CreateBuffer(&matrixBufferDesc, NULL, &MatrixBuffer))) return false;

	g = 9.81f;
	N = 64;
	Nplus1 = N + 1;
	A = 0.0005f;
	w = XMFLOAT2(32.0f, 32.0f);
	length = 64.0f;
	time = 0.0f;

	h_tilde = new complex<float>[N*N];
	h_tilde_slopex = new complex<float>[N*N];
	h_tilde_slopez = new complex<float>[N*N];
	h_tilde_dx = new complex<float>[N*N];
	h_tilde_dz = new complex<float>[N*N];
	vertices.resize(Nplus1*Nplus1);
	indices.reserve(Nplus1*Nplus1 * 6);

	fft = new cFFT(N);

	distribution = normal_distribution<float>(0.0f, 1.0f);

	int index;
	indices_count = 0;

	complex<float> htilde0, htilde0mk_conj;
	for (int m_prime = 0; m_prime < Nplus1; ++m_prime)
	{
		for (int n_prime = 0; n_prime < Nplus1; ++n_prime)
		{
			index = m_prime*Nplus1 + n_prime;

			// vertex
			htilde0 = hTilde_0(n_prime, m_prime);
			htilde0mk_conj = conj(hTilde_0(-n_prime, -m_prime));

			vertices[index].a = real(htilde0);
			vertices[index].b = imag(htilde0);
			vertices[index]._a = real(htilde0mk_conj);
			vertices[index]._b = imag(htilde0mk_conj);

			vertices[index].ox = vertices[index].x = (n_prime - N / 2.0f)*length / N;
			vertices[index].oy = vertices[index].y = 0.0f;
			vertices[index].oz = vertices[index].z = (m_prime - N / 2.0f)*length / N;

			vertices[index].nx = 0.0f;
			vertices[index].ny = 1.0f;
			vertices[index].nz = 0.0f;
		}
	}

	BuildQuadPatchIB(device);
	BuildQuadPatchVB(device);

	D3D11_RASTERIZER_DESC rastDesc;
	ZeroMemory(&rastDesc, sizeof(D3D11_RASTERIZER_DESC));
	rastDesc.FillMode = D3D11_FILL_WIREFRAME;
	rastDesc.CullMode = D3D11_CULL_NONE;
	rastDesc.FrontCounterClockwise = false;
	rastDesc.DepthClipEnable = false;

	if (FAILED(device->CreateRasterizerState(&rastDesc, &mRastStateFrame))) return false;

	D3DX11_FFT_DESC  fftdesc = {};
	fftdesc.ElementLengths[0] = fftdesc.ElementLengths[1] = fftdesc.ElementLengths[2] = 1;

	fftdesc.NumDimensions = 1;
	fftdesc.ElementLengths[0] = 102;
	fftdesc.DimensionMask = D3DX11_FFT_DIM_MASK_1D;
	fftdesc.Type = D3DX11_FFT_DATA_TYPE_COMPLEX;

	D3DX11_FFT_BUFFER_INFO  fftbufferinfo = {};
	mFFTDevice = 0;
	HRESULT hr;
	if (FAILED(hr = D3DX11CreateFFT((ID3D11DeviceContext*)dc, &fftdesc, 0, &fftbufferinfo, &mFFTDevice)))
	//if FAILED(hr = D3DX11CreateFFT2DComplex(dc, 1, 1, 0, &FFTdesc, &mFFTDevice))
	{
		return false;
	}

	return true;
}

void WaterClass::Draw(ID3D11DeviceContext1 * mImmediateContext, std::shared_ptr<CameraClass> Camera)
{
	XMMATRIX ViewProjTrans = Camera->GetViewProjTransMatrix();

	UINT stride = sizeof(Vertex);
	UINT offset = 0;

	mImmediateContext->PSSetShader(mPixelShader, NULL, 0);

	D3D11_MAPPED_SUBRESOURCE mappedResources;
	MatrixBufferType *dataPtr;

	mImmediateContext->Map(MatrixBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResources);

	dataPtr = (MatrixBufferType*)mappedResources.pData;

	dataPtr->gWorld = ViewProjTrans*XMMatrixTranspose(XMMatrixTranslation(0.0f, 15.0f, 0.0f));

	mImmediateContext->Unmap(MatrixBuffer, 0);

	mImmediateContext->Map(mQuadPatchVB, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResources);
	Vertex* v = reinterpret_cast<Vertex*>(mappedResources.pData);
	for (int i = 0; i < Nplus1*Nplus1; ++i)
	{
		v[i] = Vertex{ XMFLOAT3(vertices[i].x, vertices[i].y, vertices[i].z),
					   XMFLOAT3(vertices[i].nx, vertices[i].ny, vertices[i].nz),
					   XMFLOAT3(vertices[i].a, vertices[i].b, vertices[i].c),
					   XMFLOAT3(vertices[i]._a, vertices[i]._b, vertices[i]._c),
					   XMFLOAT3(vertices[i].ox, vertices[i].oy, vertices[i].oz) };
	}

	mImmediateContext->Unmap(mQuadPatchVB, 0);

	mImmediateContext->VSSetConstantBuffers(0, 1, &MatrixBuffer);

	mImmediateContext->VSSetShader(mVertexShader, NULL, 0);

	mImmediateContext->IASetInputLayout(mInputLayout);

	mImmediateContext->IASetIndexBuffer(mQuadPatchIB, DXGI_FORMAT_R32_UINT, 0);
	mImmediateContext->IASetVertexBuffers(0, 1, &mQuadPatchVB, &stride, &offset);
	mImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	mImmediateContext->RSSetState(mRastStateFrame);

	mImmediateContext->DrawIndexed(indices_count, 0, 0);
}

void WaterClass::DrawWithShadow(ID3D11DeviceContext1 * mImmediateContext, std::shared_ptr<CameraClass> Camera, ID3D11ShaderResourceView * shadowmap)
{
	XMMATRIX ViewProjTrans = Camera->GetViewProjTransMatrix();

	UINT stride = sizeof(Vertex);
	UINT offset = 0;

	mImmediateContext->PSSetShader(mPixelShader, NULL, 0);
	mImmediateContext->PSSetShaderResources(0, 1, &shadowmap);

	D3D11_MAPPED_SUBRESOURCE mappedResources;
	MatrixBufferType *dataPtr;

	mImmediateContext->Map(MatrixBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResources);

	dataPtr = (MatrixBufferType*)mappedResources.pData;

	dataPtr->gWorld = ViewProjTrans*XMMatrixTranspose(XMMatrixTranslation(0.0f, 15.0f, 0.0f));

	mImmediateContext->Unmap(MatrixBuffer, 0);

	mImmediateContext->VSSetConstantBuffers(0, 1, &MatrixBuffer);

	mImmediateContext->VSSetShader(mVertexShader, NULL, 0);

	mImmediateContext->IASetInputLayout(mInputLayout);

	mImmediateContext->IASetIndexBuffer(mQuadPatchIB, DXGI_FORMAT_R16_UINT, 0);
	mImmediateContext->IASetVertexBuffers(0, 1, &mQuadPatchVB, &stride, &offset);
	mImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

	mImmediateContext->DrawIndexed(4, 0, 0);
}

float WaterClass::dispersion(int n_prime, int m_prime)
{
	float w_0 = 2.0f*XM_PI / 200.0f;
	float kx = XM_PI*(2 * n_prime - N) / length;
	float kz = XM_PI*(2 * m_prime - N) / length;
	return floor(sqrt(g*sqrt(kx*kx + kz*kz)) / w_0)*w_0;
}

float WaterClass::phillips(int n_prime, int m_prime)
{
	XMFLOAT2 k = { XM_PI*(2 * n_prime - N) / length, XM_PI*(2 * m_prime - N) / length };
	float k_length = sqrt(k.x*k.x + k.y*k.y);
	if (k_length < 0.000001) return 0.0f;

	float k_length2 = k_length*k_length;
	float k_length4 = k_length2*k_length2;
	float w_length = sqrt(w.x*w.x + w.y*w.y);

	float k_dot_w = (k.x / k_length)*(w.x / w_length) + (k.y / k_length)*(w.y / w_length);
	float k_dot_w2 = k_dot_w*k_dot_w;

	float L = w_length*w_length / g;
	float L2 = L*L;

	float damping = 0.001f;
	float l2 = L2*damping*damping;

	return A * exp(-1.0f / (k_length2*L2)) / k_length4*k_dot_w2*exp(-k_length2*l2);
}

complex<float> WaterClass::hTilde_0(int n_prime, int m_prime)
{
	float x1, x2, w;
	do {
		x1 = 2.0f*distribution(generator) - 1.0f;
		x2 = 2.0f*distribution(generator) - 1.0f;
		w = x1*x1 + x2*x2;
	} while (w >= 1.0f);
	w = sqrt((-1.0f*log(w)) / w);
	complex<float> r = { x1*w, x2*w };
	return r * static_cast<float>(sqrt(phillips(n_prime, m_prime) / 2.0f));
}

complex<float> WaterClass::hTilde(float t, int n_prime, int m_prime)
{
	int index = m_prime*Nplus1 + n_prime;

	complex<float> htilde0(vertices[index].a, vertices[index].b);
	complex<float> htilde0mkconj(vertices[index]._a, vertices[index]._b);

	float omegat = dispersion(n_prime, m_prime)*t;

	float cos_ = cos(omegat);
	float sin_ = sin(omegat);

	complex<float> c0(cos_, sin_);
	complex<float> c1(cos_, -sin_);

	return htilde0*c0 + htilde0mkconj*c1;
}

complex_vector_normal WaterClass::h_D_and_n(XMFLOAT2 x, float t)
{
	complex<float> h(0.0f, 0.0f);
	XMFLOAT2 D(0.0f, 0.0f);
	XMFLOAT3 n(0.0f, 0.0f, 0.0f);

	complex<float> c, res, htilde_c;
	XMFLOAT2 k;
	float kx, kz, k_length, k_dot_x;

	for (int m_prime = 0; m_prime < N; ++m_prime)
	{
		kz = 2.0f*XM_PI*(m_prime - N / 2.0f) / length;
		for (int n_prime = 0; n_prime < N; ++n_prime)
		{
			kx = 2.0f*XM_PI*(n_prime - N / 2.0f) / length;
			k = { kx, kz };

			k_length = sqrt(kx*kx + kz*kz);
			k_dot_x = kx*x.x + kz*x.y;

			c = complex<float>(cos(k_dot_x), sin(k_dot_x));
			htilde_c = hTilde(t, n_prime, m_prime)*c;

			h += htilde_c;

			n.x += -kx*imag(htilde_c);
			n.z += -kz*imag(htilde_c);

			if (k_length < 0.000001) continue;
			D.x += kx / k_length*imag(htilde_c);
			D.y += kz / k_length*imag(htilde_c);
		}
	}

	float n_length = sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
	n.x = (0.0f - n.x) / n_length;
	n.y = (1.0f - n.y) / n_length;
	n.z = (0.0f - n.z) / n_length;

	complex_vector_normal cvn;
	cvn.h = h;
	cvn.D = D;
	cvn.n = n;

	return cvn;
}

void WaterClass::evaluateWaves(float t)
{
	t = time += t;
	float lambda = -1.0f;
	int index;
	XMFLOAT2 x;
	XMFLOAT2 d;
	complex_vector_normal h_d_and_n;
	for (int m_prime = 0; m_prime < N; ++m_prime)
	{
		for (int n_prime = 0; n_prime < N; ++n_prime)
		{
			index = m_prime*Nplus1 + n_prime;

			x = XMFLOAT2(vertices[index].x, vertices[index].z);

			h_d_and_n = h_D_and_n(x, t);

			vertices[index].y = real(h_d_and_n.h);

			vertices[index].x = vertices[index].ox + lambda*h_d_and_n.D.x;
			vertices[index].z = vertices[index].oz + lambda*h_d_and_n.D.y;

			vertices[index].nx = h_d_and_n.n.x;
			vertices[index].ny = h_d_and_n.n.y;
			vertices[index].nz = h_d_and_n.n.z;

			if (n_prime == 0 && m_prime == 0) {
				vertices[index + N + Nplus1 * N].y = real(h_d_and_n.h);

				vertices[index + N + Nplus1 * N].x = vertices[index + N + Nplus1 * N].ox + lambda*h_d_and_n.D.x;
				vertices[index + N + Nplus1 * N].z = vertices[index + N + Nplus1 * N].oz + lambda*h_d_and_n.D.y;

				vertices[index + N + Nplus1 * N].nx = h_d_and_n.n.x;
				vertices[index + N + Nplus1 * N].ny = h_d_and_n.n.y;
				vertices[index + N + Nplus1 * N].nz = h_d_and_n.n.z;
			}
			if (n_prime == 0) {
				vertices[index + N].y = real(h_d_and_n.h);

				vertices[index + N].x = vertices[index + N].ox + lambda*h_d_and_n.D.x;
				vertices[index + N].z = vertices[index + N].oz + lambda*h_d_and_n.D.y;

				vertices[index + N].nx = h_d_and_n.n.x;
				vertices[index + N].ny = h_d_and_n.n.y;
				vertices[index + N].nz = h_d_and_n.n.z;
			}
			if (m_prime == 0) {
				vertices[index + Nplus1 * N].y = real(h_d_and_n.h);

				vertices[index + Nplus1 * N].x = vertices[index + Nplus1 * N].ox + lambda*h_d_and_n.D.x;
				vertices[index + Nplus1 * N].z = vertices[index + Nplus1 * N].oz + lambda*h_d_and_n.D.y;

				vertices[index + Nplus1 * N].nx = h_d_and_n.n.x;
				vertices[index + Nplus1 * N].ny = h_d_and_n.n.y;
				vertices[index + Nplus1 * N].nz = h_d_and_n.n.z;
			}
		}
	}
}

void WaterClass::evaluateWavesFFT(float t)
{
	t = time += t;
	float kx, kz, len, lambda = -1.0f;
	int index, index1;

	for (int m_prime = 0; m_prime < N; ++m_prime)
	{
		kz = XM_PI * (2.0f*m_prime - N) / length;
		for (int n_prime = 0; n_prime < N; ++n_prime)
		{
			kx = XM_PI*(2.0f * n_prime - N) / length;
			len = sqrt(kx*kx + kz*kz);
			index = m_prime*N + n_prime;

			h_tilde[index] = hTilde(t, n_prime, m_prime);
			h_tilde_slopex[index] = h_tilde[index] * complex<float>(0, kx);
			h_tilde_slopez[index] = h_tilde[index] * complex<float>(0, kz);
			if (len < 0.000001f)
			{
				h_tilde_dx[index] = complex<float>(0.0f, 0.0f);
				h_tilde_dz[index] = complex<float>(0.0f, 0.0f);
			}
			else
			{
				h_tilde_dx[index] = h_tilde[index] * complex<float>(0, -kx / len);
				h_tilde_dz[index] = h_tilde[index] * complex<float>(0, -kz / len);
			}
		}
	}

	for (int m_prime = 0; m_prime < N; ++m_prime)
	{
		fft->fft(h_tilde, h_tilde, 1, m_prime*N);
		//fft->fft(h_tilde_slopex, h_tilde_slopex, 1, m_prime*N);
		//fft->fft(h_tilde_slopez, h_tilde_slopez, 1, m_prime*N);
		//fft->fft(h_tilde_dx, h_tilde_dx, 1, m_prime*N);
		//fft->fft(h_tilde_dz, h_tilde_dz, 1, m_prime*N);
	}
	for (int n_prime = 0; n_prime < N; ++n_prime)
	{
		fft->fft(h_tilde, h_tilde, N, n_prime);
		//fft->fft(h_tilde_slopex, h_tilde_slopex, N, n_prime);
		//fft->fft(h_tilde_slopez, h_tilde_slopez, N, n_prime);
		//fft->fft(h_tilde_dx, h_tilde_dx, N, n_prime);
		//fft->fft(h_tilde_dz, h_tilde_dz, N, n_prime);
	}

	float sign;
	float signs[] = { 1.0f, -1.0f };
	XMFLOAT3 n;
	for (int m_prime = 0; m_prime < N; m_prime++)
	{
		for (int n_prime = 0; n_prime < N; ++n_prime)
		{
			index = m_prime*N + n_prime;
			index1 = m_prime*Nplus1 + n_prime;

			sign = signs[(n_prime + m_prime) & 1];

			h_tilde[index] = h_tilde[index] * sign;

			//height
			vertices[index1].y = real(h_tilde[index]);

			// displacement
			h_tilde_dx[index] = h_tilde_dx[index] * sign;
			h_tilde_dz[index] = h_tilde_dz[index] * sign;
			vertices[index1].x = vertices[index1].ox + real(h_tilde_dx[index]) * lambda;
			vertices[index1].z = vertices[index1].oz + real(h_tilde_dz[index]) * lambda;

			//normal
			h_tilde_slopex[index] = h_tilde_slopex[index] * sign;
			h_tilde_slopez[index] = h_tilde_slopez[index] * sign;
			float n_length = sqrt(real(h_tilde_slopex[index])*real(h_tilde_slopex[index]) + 1 + real(h_tilde_slopez[index])*real(h_tilde_slopez[index]));
			XMFLOAT3 n = { -real(h_tilde_slopex[index]) / n_length, 1.0f / n_length, -real(h_tilde_slopez[index]) / n_length };
			vertices[index1].nx = n.x;
			vertices[index1].ny = n.y;
			vertices[index1].nz = n.z;

			// for tiling
			if (n_prime == 0 && m_prime == 0) {
				vertices[index1 + N + Nplus1 * N].y = real(h_tilde[index]);

				vertices[index1 + N + Nplus1 * N].x = vertices[index1 + N + Nplus1 * N].ox + real(h_tilde_dx[index]) * lambda;
				vertices[index1 + N + Nplus1 * N].z = vertices[index1 + N + Nplus1 * N].oz + real(h_tilde_dz[index]) * lambda;

				vertices[index1 + N + Nplus1 * N].nx = n.x;
				vertices[index1 + N + Nplus1 * N].ny = n.y;
				vertices[index1 + N + Nplus1 * N].nz = n.z;
			}
			if (n_prime == 0) {
				vertices[index1 + N].y = real(h_tilde[index]);

				vertices[index1 + N].x = vertices[index1 + N].ox + real(h_tilde_dx[index]) * lambda;
				vertices[index1 + N].z = vertices[index1 + N].oz + real(h_tilde_dz[index]) * lambda;

				vertices[index1 + N].nx = n.x;
				vertices[index1 + N].ny = n.y;
				vertices[index1 + N].nz = n.z;
			}
			if (m_prime == 0) {
				vertices[index1 + Nplus1 * N].y = real(h_tilde[index]);

				vertices[index1 + Nplus1 * N].x = vertices[index1 + Nplus1 * N].ox + real(h_tilde_dx[index]) * lambda;
				vertices[index1 + Nplus1 * N].z = vertices[index1 + Nplus1 * N].oz + real(h_tilde_dz[index]) * lambda;

				vertices[index1 + Nplus1 * N].nx = n.x;
				vertices[index1 + Nplus1 * N].ny = n.y;
				vertices[index1 + Nplus1 * N].nz = n.z;
			}
		}
	}
}

void WaterClass::evaluateWavesGPU(float t)
{

}

void WaterClass::BuildQuadPatchVB(ID3D11Device1 * device)
{
	vector<Vertex> patchVertices;

	for (int i = 0; i < Nplus1*Nplus1; ++i)
	{
		patchVertices.push_back(Vertex{ XMFLOAT3(vertices[i].x, vertices[i].y, vertices[i].z),
										XMFLOAT3(vertices[i].nx, vertices[i].ny, vertices[i].nz),
										XMFLOAT3(vertices[i].a, vertices[i].b, vertices[i].c),
										XMFLOAT3(vertices[i]._a, vertices[i]._b, vertices[i]._c),
										XMFLOAT3(vertices[i].ox, vertices[i].oy, vertices[i].oz) });
	}

	D3D11_BUFFER_DESC vbd;
	vbd.Usage = D3D11_USAGE_DYNAMIC;
	vbd.ByteWidth = sizeof(Vertex)*patchVertices.size();
	vbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	vbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	vbd.MiscFlags = 0;
	vbd.StructureByteStride = 0;

	D3D11_SUBRESOURCE_DATA vinitData;
	vinitData.pSysMem = &patchVertices[0];
	device->CreateBuffer(&vbd, &vinitData, &mQuadPatchVB);


}

bool WaterClass::BuildQuadPatchIB(ID3D11Device1 * device)
{
	vector<UINT> indices;

	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			indices.push_back(i*Nplus1 + j);
			indices.push_back((i + 1)*Nplus1 + j);
			indices.push_back((i + 1)*Nplus1 + j + 1);
			indices.push_back(i*Nplus1 + j);
			indices.push_back((i + 1)*Nplus1 + j + 1);
			indices.push_back(i*Nplus1 + j + 1);
		}
	}

	indices_count = indices.size();

	D3D11_BUFFER_DESC ibd;
	ibd.Usage = D3D11_USAGE_IMMUTABLE;
	ibd.ByteWidth = sizeof(UINT)*indices.size();
	ibd.BindFlags = D3D11_BIND_INDEX_BUFFER;
	ibd.CPUAccessFlags = 0;
	ibd.MiscFlags = 0;
	ibd.StructureByteStride = 0;

	D3D11_SUBRESOURCE_DATA iinitData;
	iinitData.pSysMem = &indices[0];
	if (FAILED(device->CreateBuffer(&ibd, &iinitData, &mQuadPatchIB))) return false;

	return true;
}

bool WaterClass::CreateInputLayoutAndShaders(ID3D11Device1 * device)
{
	ifstream stream;
	size_t size;
	char* data;

	// pixel
	stream.open("..\\Debug\\WaterPS.cso", ifstream::binary);
	if (stream.good())
	{
		stream.seekg(0, ios::end);
		size = size_t(stream.tellg());
		data = new char[size];
		stream.seekg(0, ios::beg);
		stream.read(&data[0], size);
		stream.close();

		if (FAILED(device->CreatePixelShader(data, size, 0, &mPixelShader)))
		{
			LogError(L"Failed to create water pixel shader");
			return false;
		}
		delete[] data;
	}
	else
	{
		LogError(L"Failed to open WaterPS.cso");
		return false;
	}

	LogSuccess(L"Water pixel shader created.");

	// vertex shader
	stream.open("..\\Debug\\WaterVS.cso", ifstream::binary);
	if (stream.good())
	{
		stream.seekg(0, ios::end);
		size = size_t(stream.tellg());
		data = new char[size];
		stream.seekg(0, ios::beg);
		stream.read(&data[0], size);
		stream.close();

		if (FAILED(device->CreateVertexShader(data, size, 0, &mVertexShader)))
		{
			LogError(L"Failed to create water vertex shader");
			return false;
		}
	}
	else
	{
		LogError(L"Fail to open WaterPS.cso");
		return false;
	}

	LogSuccess(L"Water vertex shader created.");

	D3D11_INPUT_ELEMENT_DESC vertexDesc[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "TEXCOORD", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "TEXCOORD", 1, DXGI_FORMAT_R32G32B32_FLOAT, 0, 36, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "POSITION", 1, DXGI_FORMAT_R32G32B32_FLOAT, 0, 48, D3D11_INPUT_PER_VERTEX_DATA, 0 }
	};

	int numElements = sizeof(vertexDesc) / sizeof(vertexDesc[0]);

	if (FAILED(device->CreateInputLayout(vertexDesc, numElements, data, size, &mInputLayout))) return false;

	delete[] data;

	return true;
}
