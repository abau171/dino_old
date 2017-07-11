# Dino
Path-tracing GPU rendering engine.

### Demos and Example Renders

[Test Scene Demo Video](https://www.youtube.com/watch?v=fVtnDnm3HdQ)

[Dragon Demo Video](https://www.youtube.com/watch?v=iqLyLICHI1Y)

![Example 1](dino/example_renders/colonial_1800_6m.png)
![Example 2](dino/example_renders/focused_caustics_4500_6m.png)
![Example 3](dino/example_renders/attenuation_teapot_1400_5m.png)
![Example 4](dino/example_renders/cloudy_dragon_1700_18m.png)

### About

Dino was developed as a final project developed for RIT's Global Illumination class. It is a toy implementation of the path tracing algorithm from Kajiya's The Rendering Equation paper. It has been extended to include visual effects from Cook et al.'s Distributed Ray Tracing paper (with modifications to work with path tracing. Rendering happens entirely on a GPU using Nvidia's CUDA API, and images are progressively refined and displayed in an interactive window.

Although all code was written from scratch, Dino's design was influenced by:
- [Peter and Karl's GPU Path Tracer](http://gpupathtracer.blogspot.com/)
- [Ray Tracey's GPU Path Tracing Blog Series](http://raytracey.blogspot.com/2015/10/gpu-path-tracing-tutorial-1-drawing.html)
- [Blender Cycles Render Engine](https://docs.blender.org/manual/en/dev/render/cycles/)

### Features

- unidirectional path Tracing
- sphere primitives, and basic .obj file loading
- BVH spatial partitioning structures for models
- Phong PDFs with importance sampling for diffuse and specular shading
- perfect reflection, glossy reflection, transparency, translucency, refraction
- Beer's law attenuation
- subsurface scattering with Henyey-Greenstein phase function
- depth of field
- interactive camera

### Controls

- W, A, S, D keys for basic camera movement
- R and F move the camera up and down
- T and G update the focal distance (Depth of Field)
- Alternatively, right clicking in the scene updates the focal distance to the selected object.
- Y and H change the camera aperture size
- C resets the render buffer
- P pauses the renderer
- O saves the rendered image, prompting for a file name
- Shift + O saves the rendered image with a timestamp as the file name.
- Q quits the application

### Models Used

- Standford Dragon: http://graphics.stanford.edu/data/3Dscanrep/
- Watertight Utah Teapot: http://goanna.cs.rmit.edu.au/~pknowles/models.html
- Colonial Buildings: https://sketchfab.com/
