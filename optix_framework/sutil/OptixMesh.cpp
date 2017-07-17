/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */


#include "OptixMesh.h"
#include "OptixMeshClasses.h"
#include "OptixMeshImpl.h"

#include <optixpp_namespace.h>

using namespace OptixMeshClasses;


OptixMesh::OptixMesh( optix::Context context,
                      optix::GeometryGroup geometrygroup,
                      const char* ASBuilder,
                      const char* ASTraverser,
                      const char* ASRefine,
                      bool large_geom )
: m_impl( new OptixMeshImpl( context,
                             geometrygroup,
                             ASBuilder,
                             ASTraverser,
                             ASRefine,
                             large_geom ) )
{ }

OptixMesh::OptixMesh( optix::Context context,
                      optix::GeometryGroup geometrygroup,
                      optix::Material default_material,
                      const char* ASBuilder,
                      const char* ASTraverser,
                      const char* ASRefine,
                      bool large_geom )
: m_impl( new OptixMeshImpl( context,
                             geometrygroup,
                             default_material,
                             ASBuilder,
                             ASTraverser,
                             ASRefine,
                             large_geom ) )
{ }

OptixMesh::~OptixMesh() { if( m_impl ) delete m_impl; }


void OptixMesh::loadBegin_Geometry( const std::string& filename ) { m_impl->beginLoad(filename); }
void OptixMesh::loadFinish_Materials() { m_impl->finalizeLoad(); }

void OptixMesh::setDefaultIntersectionProgram( optix::Program inx_program ) {
  m_impl->setDefaultIntersectionProgram( inx_program );
}
optix::Program OptixMesh::getDefaultIntersectionProgram() const {
  return m_impl->getDefaultIntersectionProgram();
}

void OptixMesh::setDefaultBoundingBoxProgram( optix::Program bbox_program ) {
  m_impl->setDefaultBoundingBoxProgram( bbox_program );
}
optix::Program OptixMesh::getDefaultBoundingBoxProgram() const {
  return m_impl->getDefaultBoundingBoxProgram();
}

void OptixMesh::setDefaultOptixMaterial( optix::Material material ) {
  m_impl->setDefaultOptixMaterial( material );
}
optix::Material OptixMesh::getDefaultOptixMaterial() const {
  return m_impl->getDefaultOptixMaterial();
}

void OptixMesh::setDefaultMaterialParams( const MeshMaterialParams& params ) {
  m_impl->setDefaultMaterialParams( params );
}
const MeshMaterialParams& OptixMesh::getDefaultMaterialParams() const {
  return m_impl->getDefaultMaterialParams();
}

int OptixMesh::getNumVertices() const { return m_impl->getNumVertices(); }
int OptixMesh::getNumNormals() const { return m_impl->getNumNormals(); }
int OptixMesh::getNumColors() const { return m_impl->getNumColors(); }
int OptixMesh::getNumTextureCoordinates() const { return m_impl->getNumTextureCoordinates(); }

int OptixMesh::getNumTriangles() const { return m_impl->getNumTriangles(); }

float* OptixMesh::getVertexData() { return m_impl->getVertexData(); }
const float* OptixMesh::getVertexData() const { return m_impl->getVertexData(); }

float* OptixMesh::getNormalData() { return m_impl->getNormalData(); }
const float* OptixMesh::getNormalData() const { return m_impl->getNormalData(); }

unsigned char* OptixMesh::getColorData() { return m_impl->getColorData(); }
const unsigned char* OptixMesh::getColorData() const { return m_impl->getColorData(); }

float* OptixMesh::getTextureCoordinateData() { return m_impl->getTextureCoordinateData(); }
const float* OptixMesh::getTextureCoordinateData() const { return m_impl->getTextureCoordinateData(); }

int OptixMesh::getVertexStride() const { return m_impl->getVertexStride(); }
int OptixMesh::getNormalStride() const { return m_impl->getNormalStride(); }
int OptixMesh::getColorStride() const { return m_impl->getColorStride(); }
int OptixMesh::getTextureCoordinateStride() const { return m_impl->getTextureCoordinateStride(); }

const float* OptixMesh::getBBoxMin() const { return m_impl->getBBoxMin(); }
const float* OptixMesh::getBBoxMax() const { return m_impl->getBBoxMax(); }

void OptixMesh::updateBBox() { m_impl->updateBBox(); }

const std::string& OptixMesh::getMaterialLibraryName() const {
  return m_impl->getMaterialLibraryName();
}

MeshGroup& OptixMesh::getMeshGroup(const std::string& group_name) {
  return m_impl->getMeshGroup(group_name);
}

const MeshGroup& OptixMesh::getMeshGroup(const std::string& group_name) const {
  return m_impl->getMeshGroup(group_name);
}

GroupGeometryInfo& OptixMesh::getGroupGeometryInfo( const std::string& group_name ) {
  return m_impl->getGroupGeometryInfo(group_name);
}

const GroupGeometryInfo& OptixMesh::getGroupGeometryInfo( const std::string& group_name ) const {
  return m_impl->getGroupGeometryInfo(group_name);
}

size_t OptixMesh::getMaterialCount() const { return m_impl->getMaterialCount(); }

void OptixMesh::setMeshMaterialParams( int i, const MeshMaterialParams& params ) {
  m_impl->setMeshMaterialParams( i, params );
}

MeshMaterialParams& OptixMesh::getMeshMaterialParams( int i ) {
  return m_impl->getMeshMaterialParams( i );
}

const MeshMaterialParams& OptixMesh::getMeshMaterialParams( int i ) const {
  return m_impl->getMeshMaterialParams( i );
}

void OptixMesh::setOptixMaterial( int i, optix::Material material ) {
  m_impl->setOptixMaterial( i, material );
}

const optix::Material& OptixMesh::getOptixMaterial( int i ) const {
  return m_impl->getOptixMaterial( i );
}


optix::Aabb OptixMesh::getSceneBBox() const { return m_impl->getSceneBBox(); }
optix::Buffer OptixMesh::getLightBuffer() const { return m_impl->getLightBuffer(); }

const optix::Matrix4x4& OptixMesh::getLoadingTransform() const {
  return m_impl->getLoadingTransform();
}
  
void OptixMesh::setLoadingTransform( const optix::Matrix4x4& transform ) {
  m_impl->setLoadingTransform( transform );
}


void OptixMesh::addMaterial() {
  m_impl->addMaterial();
}

int OptixMesh::getGroupMaterialNumber(const MeshGroup& group) const {
  return m_impl->getGroupMaterialNumber( group );
}


void OptixMesh::setOptixInstanceMatParams( optix::GeometryInstance gi,
                                           const MeshMaterialParams& params ) const
{
  m_impl->setOptixInstanceMatParams( gi, params );
}
