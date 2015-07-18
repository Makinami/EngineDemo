#pragma once

#include <d3d11_1.h>
#include <fstream>
#include <string>

bool LoadShader(_In_ std::wstring fileName, _Out_ char* &data, _Out_ size_t &size);

bool CreatePSFromFile(_In_ std::wstring fileName, ID3D11Device1 * device, ID3D11PixelShader* &ps);

bool CreateCSFromFile(_In_ std::wstring fileName, ID3D11Device1 * device, ID3D11ComputeShader* &cs);

bool CreateDSFromFile(_In_ std::wstring fileName, ID3D11Device1 * device, ID3D11DomainShader* &ds);

bool CreateHSFromFile(_In_ std::wstring fileName, ID3D11Device1 * device, ID3D11HullShader* &hs);

bool CreateVSFromFile(_In_ std::wstring fileName, ID3D11Device1 * device, ID3D11VertexShader* &vs);

bool CreateGSFromFile(_In_ std::wstring fileName, ID3D11Device1 * device, ID3D11GeometryShader* &gs);

bool CreateVSAndInputLayout(_In_ std::wstring fileName, ID3D11Device1 * device, ID3D11VertexShader* &vs, const D3D11_INPUT_ELEMENT_DESC* ilDesc, UINT numElements, ID3D11InputLayout* &il);