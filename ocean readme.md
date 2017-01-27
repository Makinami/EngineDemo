# Ocean

## Mesh

My own simplified implementation of Filip Strugar's [CDLOD].
* VS -> HS -> DS -> PS pipeline
  * **Vertex Shader**  
    First morphing vertex grid between LODs, scalling and moving acording to position (Pos.xyz).  
    Than read gradient field, and based on that calculate phase (params.x), depth dependency (params.y), direction to a coast (params.zw).  
  * **Hull Shader**  
    Tesselation based on depth dependency.
  * **Domain Shader**  
    Read opean sea displacement.  
    From displacement noise and wind direction parameter calculate amplitude scaling factor.  
    Amplitude calculate as half of the significant wave height: $A=\frac{1}{2}\frac{0.27*|\bold{w}|^2}{g}$. From this and phase from VS we calculate gernsten waves shape, which is interpolated with open ocean shape with depth dependency (VS) as parameter.  
    Out:  
    * PosH : SV_POSITION
    * PosF : xy - world space flat position, z - wave height excursion, w - amplitude's factor
    * PosW : xyz - world space position
    * params : x - phase, y - depth, zw - direction to a coast
  * **Pixel Shader**  
    Analitic normals for gernsten, open ocean normals from slope map.

## Lighting
Bruneton's lighting model extended by transparency (channel dependent linear light absorption), and screen space refractions. +foam with ramp texture
$$waterLight = lerp(seaLight, seenBottom, 1-extinction)*(foam*0.05 / PI + (1.0 - foam)*(1.0 - fresnelUp))$$ (1)
$$skyLight = skyRadiance*(foam + (1.0 - foam)*fresnelUp)$$ (2)
$$sunLight = Lsun * (foam + (1.0 - foam)*reflectedRadiance)$$ (3)


[CDLOD]: http://www.vertexasylum.com/downloads/cdlod/cdlod_latest.pdf