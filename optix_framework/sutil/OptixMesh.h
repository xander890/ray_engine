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

#ifndef __samples_util_optix_mesh_h__
#define __samples_util_optix_mesh_h__

#include "OptixMeshImpl.h"

#include "sutilapi.h"

#include <optixpp_namespace.h>
#include <optixu_aabb_namespace.h>
#include <optixu_matrix_namespace.h>

#include <string>

class MeshGroup;
class MeshMaterialParams;
struct OptixMeshMaterial;


//-----------------------------------------------------------------------------
// 
//  OptixMesh class declaration 
//
//-----------------------------------------------------------------------------
class SUTILCLASSAPI OptixMesh
{
public:

  SUTILAPI OptixMesh( optix::Context context,               // Context for RT object creation
                      optix::GeometryGroup geometrygroup,   // Empty geom group to hold model
                      const char* ASBuilder   = "Sbvh",
                      const char* ASTraverser = "Bvh",
                      const char* ASRefine = "0",
                      bool large_geom = false );            // Reorder geom data for page fault minimization

  SUTILAPI OptixMesh( optix::Context context,               // Context for RT object creation
                      optix::GeometryGroup geometrygroup,   // empty geom group to hold model
                      optix::Material default_material,     // default OptiX material to assign
                      const char* ASBuilder   = "Sbvh",
                      const char* ASTraverser = "Bvh",
                      const char* ASRefine = "0",
                      bool large_geom = false );            // Reorder geom data for page fault minimization

  SUTILAPI ~OptixMesh();

  /**
   * Calls functor( group ) for each group in the mesh, where 'group' is of
   * the type MeshGroup.
   */
  template <class Functor>
  void forEachGroup( Functor functor ) const;

  template <class Functor>
  void forEachGroup( Functor functor );


  /**
   * The first pass of loading a model.  Creates central vertex data buffers
   * on the context called "vertex_buffer", "normal_buffer", and
   * "texcoord_buffer".  Creates a Geometry node for each group in the model
   * file, with OptiX index buffers named "vindex_buffer" (or "vert_idx_buffer"
   * if the large_mesh option is used), "nindex_buffer", and "tindex_buffer".
   *
   * A default OptiX Material is created with default closest_hit and any_hit
   * programs.  Default material parameters are created, unless the model file
   * supplies parameters.  The loader state is set up to assign these Materials
   * and parameters to the groups, though different Materials and parameters,
   * as well as intersection programs, may be set before calling the second
   * loading pass.
   */
  SUTILAPI void loadBegin_Geometry( const std::string& filename );

  /**
   * This method is necessary for finalizing setup of the Optix Materials and
   * GeometryInstances after beginLoad() has been called and any editing of
   * the MeshGroups, MeshMaterialParams, and optix::Materials has been finished
   * by the user.
   *
   * For each MeshGroup i in the OptixMesh, it will look up the assigned
   * material index from getMeshGroup(i), fetch the corresponding
   * optix::Geometry object from getOptixGeometryForGroup(i), and create an
   * optix::GeometryInstance using the optix::material assigned to
   * getOptixMaterialForGroup(i).  It will assign the Optix variables for the
   * GeometryInstance based on the values set in
   * getMeshMaterialParamsForGroup(i).
   */
  SUTILAPI void loadFinish_Materials();

  /**
   * These methods allow substitution of new defaults for OptixMesh's built-in
   * defaults for the Optix Material, Intersection, and Bounding Box programs,
   * as well as material parameters, that are set up by beginLoad().  These are
   * the Materials and programs that will then be attached to GeometryIntsances
   * during finalizeLoad() unless other overrides are injected before
   * finalizeLoad().
   */
  SUTILAPI void setDefaultIntersectionProgram( optix::Program inx_program );
  SUTILAPI optix::Program getDefaultIntersectionProgram() const;
  SUTILAPI void setDefaultBoundingBoxProgram( optix::Program bbox_program );
  SUTILAPI optix::Program getDefaultBoundingBoxProgram() const;
  SUTILAPI void setDefaultOptixMaterial( optix::Material material );
  SUTILAPI optix::Material getDefaultOptixMaterial() const;
  SUTILAPI void setDefaultMaterialParams( const MeshMaterialParams& params );
  SUTILAPI const MeshMaterialParams& getDefaultMaterialParams() const;

  SUTILAPI int getNumVertices() const;
  SUTILAPI int getNumNormals() const;
  SUTILAPI int getNumColors() const;
  SUTILAPI int getNumTextureCoordinates() const;

  SUTILAPI int getNumTriangles() const;

  SUTILAPI float* getVertexData();
  SUTILAPI const float* getVertexData() const;

  SUTILAPI float* getNormalData();
  SUTILAPI const float* getNormalData() const;
  
  SUTILAPI unsigned char* getColorData();
  SUTILAPI const unsigned char* getColorData() const;
  
  SUTILAPI float* getTextureCoordinateData();
  SUTILAPI const float* getTextureCoordinateData() const;

  SUTILAPI int getVertexStride() const;
  SUTILAPI int getNormalStride() const;
  SUTILAPI int getColorStride() const;
  SUTILAPI int getTextureCoordinateStride() const;

  SUTILAPI const float* getBBoxMin() const;
  SUTILAPI const float* getBBoxMax() const;

  SUTILAPI void updateBBox();

  SUTILAPI const std::string& getMaterialLibraryName() const;

  SUTILAPI MeshGroup& getMeshGroup(const std::string& group_name);
  SUTILAPI const MeshGroup& getMeshGroup(const std::string& group_name) const;

  SUTILAPI OptixMeshClasses::GroupGeometryInfo& getGroupGeometryInfo( const std::string& group_name );
  SUTILAPI const OptixMeshClasses::GroupGeometryInfo& getGroupGeometryInfo( const std::string& group_name ) const;

  SUTILAPI size_t getMaterialCount() const;
  SUTILAPI void setMeshMaterialParams( int i, const MeshMaterialParams& params );
  SUTILAPI MeshMaterialParams& getMeshMaterialParams( int i );
  SUTILAPI const MeshMaterialParams& getMeshMaterialParams( int i ) const;

  SUTILAPI void setOptixMaterial( int i, optix::Material material );
  SUTILAPI const optix::Material& getOptixMaterial( int i ) const;
  
  SUTILAPI optix::Aabb getSceneBBox() const;
  SUTILAPI optix::Buffer getLightBuffer() const;

  SUTILAPI const optix::Matrix4x4& getLoadingTransform() const;
  
  SUTILAPI void setLoadingTransform( const optix::Matrix4x4& transform );

  SUTILAPI virtual void addMaterial();
  SUTILAPI int getGroupMaterialNumber(const MeshGroup& group) const;
  SUTILAPI void setOptixInstanceMatParams( optix::GeometryInstance gi,
                                           const MeshMaterialParams& params ) const;


private:
  
  OptixMeshImpl* m_impl;
};


template <class Functor>
void OptixMesh::forEachGroup( Functor functor ) const
{
  m_impl->forEachGroup( functor );
}

template <class Functor>
void OptixMesh::forEachGroup( Functor functor )
{
  m_impl->forEachGroup( functor );
}


#endif // __samples_util_optix_mesh_h__
