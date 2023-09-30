<template>
  <div class="min-h-screen flex flex-col items-center justify-center bg-gray-100">
    
    <!-- Title -->
    <h1 class="text-4xl font-bold mb-8">Parallel Image Processing</h1>
    
    <!-- Images -->
    <div class="flex">
      <!-- Preloaded Image -->
      <img src="../../data/image_portrait.png" alt="Preloaded Image" class="mr-4 w-64 h-64">

      
      <!-- Processed Image -->
      <img v-if="processedImage" :src="processedImage" alt="Processed Image">
    </div>
    
    <!-- Buttons -->
    <div class="mt-8">
      <p class="mb-4 text-lg font-semibold">Parallel image processing options:</p>
      <button @click="processImage('color-to-grayscale')" class="bg-blue-500 text-white py-2 px-4 rounded mr-4 hover:bg-blue-600">
        Color-to-grayscale
      </button>
      <button @click="processImage('image-blur')" class="bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600">
        Image Blur
      </button>
      <button @click="processImage('reset-image')" class="bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600">
        Reset 
      </button>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      processedImage: null
    }
  },
  methods: {
    async processImage(option) {
      // Call the Flask API with the selected option
      // Update the `processedImage` data property with the returned image path
    }
  }
}
</script>

<style scoped>
img {
    width: 500px;
    height: 400px;
}
</style>
