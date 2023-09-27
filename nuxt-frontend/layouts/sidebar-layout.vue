<!-- layouts/sidebar-layout.vue -->
<template>
  <div class="sidebar-layout">
    <aside class="sidebar">
      <!-- Sidebar content goes here -->
      <nav>
        <slot name="sidebarContent"></slot>
      </nav>
    </aside>
    <main class="main-content">
      <!-- Main content goes here -->
      <h1>{{ pageTitle }}</h1>
      <slot></slot> <!-- Default slot for page content -->
    </main>
  </div>
</template>

<script>
export default {
  props: {
    pageTitle: String,
  },
}
</script>

<style scoped>
/* Add your CSS styles for the layout here */
.sidebar-layout {
  display: flex;
  flex-direction: row;
  background-color: #f0f0f0;
  padding: 20px;
}

.sidebar {
  /* Sidebar styles */
  width: 25%;
  background-color: #ffffff;
  padding: 10px;
  color: #333; /* Set the font color to dark gray (#333) */
}

.main-content {
  /* Main content styles */
  width: 75%;
  padding: 20px;
}
</style>
