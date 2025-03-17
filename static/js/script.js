document.addEventListener("DOMContentLoaded", function () {
  const modelSelectionDiv = document.getElementById("modelSelection");
  const selectAllBtn = document.getElementById("selectAll");
  const deselectAllBtn = document.getElementById("deselectAll");
  const modelSearchInput = document.getElementById("modelSearchInput");
  const clearSearchBtn = document.getElementById("clearSearch");
  const selectedModelCount = document.getElementById("selectedModelCount");
  const dropArea = document.getElementById("dropArea");
  const filesInput = document.getElementById("filesInput");
  const folderInput = document.getElementById("folderInput");
  const selectFolderBtn = document.getElementById("selectFolderBtn");
  const imagePreview = document.getElementById("imagePreview");
  const fileCounter = document.getElementById("fileCounter");
  const fileCountBadge = document.getElementById("fileCountBadge");
  const clearFilesBtn = document.getElementById("clearFiles");
  const detectBtn = document.getElementById("detectBtn");
  const uploadForm = document.getElementById("uploadForm");
  const loading = document.getElementById("loading");
  const processingStatus = document.getElementById("processingStatus");
  const progressBar = document.getElementById("progressBar");
  const progressContainer = document.getElementById("progressContainer");
  const resultContainer = document.getElementById("resultContainer");
  const batchResultsContainer = document.getElementById("batchResultsContainer");

  // Collection to store selected files
  let selectedFiles = new DataTransfer();

  // Init file selection counter
  function updateFileCounter() {
    const fileCount = selectedFiles.files.length;
    fileCountBadge.textContent = `${fileCount} file${
      fileCount !== 1 ? "s" : ""
    } selected`;
    fileCounter.classList.toggle("d-none", fileCount === 0);
    detectBtn.disabled = fileCount === 0;
  }

  // File drag & drop handling
  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  ["dragenter", "dragover"].forEach((eventName) => {
    dropArea.addEventListener(eventName, highlight, false);
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(eventName, unhighlight, false);
  });

  function highlight() {
    dropArea.classList.add("highlight");
  }

  function unhighlight() {
    dropArea.classList.remove("highlight");
  }

  // Handle dropped files
  dropArea.addEventListener("drop", handleDrop, false);

  function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
  }

  // Handle file input change
  filesInput.addEventListener("change", function () {
    handleFiles(this.files);
  });

  // Handle folder input
  selectFolderBtn.addEventListener("click", function () {
    folderInput.click();
  });

  folderInput.addEventListener("change", function () {
    handleFiles(this.files);
  });

  // Process files
  function handleFiles(fileList) {
    if (fileList.length === 0) return;

    // Show a loading message if there are many files
    if (fileList.length > 20) {
      processingStatus.textContent = `Processing ${fileList.length} files...`;
      progressContainer.style.display = "block";
    }

    let processedCount = 0;
    const totalFiles = fileList.length;
    
    for (let i = 0; i < fileList.length; i++) {
      const file = fileList[i];

      // Check if it's an image file
      if (
        !file.type.match("image.*") &&
        !file.name.match(/\.(jpg|jpeg|png|gif|bmp|tif|tiff)$/i)
      ) {
        processedCount++;
        updateProgress(processedCount, totalFiles);
        continue;
      }

      // Check if the file is already in the list (by name)
      let isDuplicate = false;
      for (let j = 0; j < selectedFiles.files.length; j++) {
        if (selectedFiles.files[j].name === file.name) {
          isDuplicate = true;
          break;
        }
      }

      if (isDuplicate) {
        processedCount++;
        updateProgress(processedCount, totalFiles);
        continue;
      }

      // Add file to the DataTransfer object
      selectedFiles.items.add(file);

      // Create preview
      const reader = new FileReader();
      reader.onload = (function (f) {
        return function (e) {
          const previewItem = document.createElement("div");
          previewItem.className = "image-preview-item";

          const img = document.createElement("img");
          img.src = e.target.result;
          img.title = f.name;

          const removeBtn = document.createElement("div");
          removeBtn.className = "remove-btn";
          removeBtn.innerHTML = "&times;";
          removeBtn.addEventListener("click", function () {
            removeFile(f.name);
            previewItem.remove();
          });

          previewItem.appendChild(img);
          previewItem.appendChild(removeBtn);
          imagePreview.appendChild(previewItem);
          
          processedCount++;
          updateProgress(processedCount, totalFiles);
        };
      })(file);

      reader.readAsDataURL(file);
    }

    // Update the input files with our collection
    updateFilesInput();
  }
  
  function updateProgress(current, total) {
    if (total <= 20) return;
    
    const percentage = Math.round((current / total) * 100);
    progressBar.style.width = `${percentage}%`;
    progressBar.textContent = `${percentage}%`;
    
    if (current >= total) {
      setTimeout(() => {
        progressContainer.style.display = "none";
        processingStatus.textContent = "";
      }, 1000);
    }
  }

  // Remove file from selection
  function removeFile(filename) {
    const dt = new DataTransfer();

    for (let i = 0; i < selectedFiles.files.length; i++) {
      const file = selectedFiles.files[i];
      if (file.name !== filename) {
        dt.items.add(file);
      }
    }

    selectedFiles = dt;
    updateFilesInput();
  }

  // Update the form's file input with selected files
  function updateFilesInput() {
    filesInput.files = selectedFiles.files;
    updateFileCounter();
  }

  // Clear all selected files
  clearFilesBtn.addEventListener("click", function () {
    selectedFiles = new DataTransfer();
    imagePreview.innerHTML = "";
    updateFilesInput();
  });

  // Function to update selected model count
  function updateSelectedModelCount() {
    const checkedCount = document.querySelectorAll(".model-checkbox:checked").length;
    const totalCount = document.querySelectorAll(".model-checkbox").length;
    selectedModelCount.textContent = `${checkedCount} of ${totalCount} selected`;
    
    // Disable detect button if no models are selected
    detectBtn.disabled = selectedFiles.files.length === 0 || checkedCount === 0;
  }

  // Helper function to detect YOLO version from filename
  function detectYoloVersion(filename) {
    filename = filename.toLowerCase();
    if (filename.includes("v12") || filename.includes("yolov12")) {
      return "v12";
    } else if (filename.includes("v11") || filename.includes("yolov11")) {
      return "v11";
    } else if (filename.includes("v10") || filename.includes("yolov10")) {
      return "v10";
    } else if (filename.includes("v9") || filename.includes("yolov9")) {
      return "v9";
    } else if (filename.includes("v8") || filename.includes("yolov8")) {
      return "v8";
    } else if (filename.includes("v5") || filename.includes("yolov5")) {
      return "v5";
    } else {
      return "other";
    }
  }

  // Helper function to detect model size from filename
  function detectModelSize(filename) {
    filename = filename.toLowerCase();
    if (filename.includes("nano") || filename.includes("tiny") || filename.includes("-n")) {
      return "size-small";
    } else if (filename.includes("large") || filename.includes("-l") || filename.includes("-x")) {
      return "size-large";
    } else {
      return "size-medium";
    }
  }

  // Function to create model cards
  function createModelCard(model, index) {
    const modelId = `model-${index}`;
    const version = detectYoloVersion(model);
    const size = detectModelSize(model);
    
    const card = document.createElement("div");
    card.className = "model-card";
    card.dataset.model = model;
    
    // Format version display
    let versionLabel = version.toUpperCase();
    if (version === "v5" || version === "v8" || version === "v10") {
      versionLabel = "YOLO" + version.toUpperCase();
    }
    
    // Format size display
    let sizeLabel = "Medium";
    if (size === "size-small") {
      sizeLabel = "Small";
    } else if (size === "size-large") {
      sizeLabel = "Large";
    }
    
    card.innerHTML = `
      <div class="model-card-header">
        <div class="form-check">
          <input class="form-check-input model-checkbox" type="checkbox" value="${model}" 
            id="${modelId}" name="selected_models[]" checked>
          <label class="form-check-label" for="${modelId}">
            ${highlightText(model, modelSearchInput.value)}
          </label>
        </div>
      </div>
      <div class="model-metadata">
        <span class="model-tag ${version}">${versionLabel}</span>
        <span class="model-tag ${size}">${sizeLabel}</span>
      </div>
    `;
    
    // Add click event to the entire card for better UX
    card.addEventListener("click", function(e) {
      // Don't trigger if clicking directly on the checkbox
      if (e.target.type !== "checkbox") {
        const checkbox = this.querySelector(".model-checkbox");
        checkbox.checked = !checkbox.checked;
        updateCardSelection(card, checkbox.checked);
        updateSelectedModelCount();
      }
    });
    
    // Add change event to checkbox
    const checkbox = card.querySelector(".model-checkbox");
    checkbox.addEventListener("change", function() {
      updateCardSelection(card, this.checked);
      updateSelectedModelCount();
    });
    
    // Set initial selection state
    updateCardSelection(card, checkbox.checked);
    
    return card;
  }
  
  // Update card selection visual state
  function updateCardSelection(card, isSelected) {
    if (isSelected) {
      card.classList.add("selected");
    } else {
      card.classList.remove("selected");
    }
  }
  
  // Helper function to highlight search text
  function highlightText(text, searchTerm) {
    if (!searchTerm) return text;
    
    const regex = new RegExp(`(${escapeRegExp(searchTerm)})`, 'gi');
    return text.replace(regex, '<span class="highlight-text">$1</span>');
  }
  
  // Escape special characters for regex
  function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }
  
  // Filter models based on search input
  function filterModels() {
    const searchTerm = modelSearchInput.value.toLowerCase();
    const cards = document.querySelectorAll(".model-card");
    let visibleCount = 0;
    
    cards.forEach(card => {
      const model = card.dataset.model.toLowerCase();
      if (model.includes(searchTerm)) {
        card.style.display = "block";
        visibleCount++;
        
        // Update the highlighted text
        const label = card.querySelector(".form-check-label");
        label.innerHTML = highlightText(card.dataset.model, searchTerm);
      } else {
        card.style.display = "none";
      }
    });
    
    // Show "no models found" message if needed
    const noModelsMsg = document.getElementById("noModelsFound");
    if (visibleCount === 0 && searchTerm !== "") {
      if (!noModelsMsg) {
        const msg = document.createElement("div");
        msg.id = "noModelsFound";
        msg.className = "no-models-found";
        msg.innerHTML = `<i class="bi bi-search"></i> No models found matching "<strong>${searchTerm}</strong>"`;
        modelSelectionDiv.appendChild(msg);
      }
    } else if (noModelsMsg) {
      noModelsMsg.remove();
    }
  }

  // Get model information and populate model selection
  fetch("/model_info")
    .then((response) => response.json())
    .then((data) => {
      if (data.models_count > 0) {
        // Create model selection cards
        modelSelectionDiv.innerHTML = "";
        const fragment = document.createDocumentFragment();
        
        // Add the models as cards
        data.available_models.forEach((model, index) => {
          const card = createModelCard(model, index);
          fragment.appendChild(card);
        });
        
        modelSelectionDiv.appendChild(fragment);
        updateSelectedModelCount();

        // Update model info section
        document.getElementById("modelInfo").innerHTML = `
          <div class="d-flex justify-content-between align-items-center">
            <h5><i class="bi bi-info-circle"></i> Models Loaded</h5>
            <span class="badge bg-success">${data.models_count} Models Available</span>
          </div>
          <p>All models are loaded and ready to use. Select which models to include in the ensemble detection.</p>
        `;
      } else {
        modelSelectionDiv.innerHTML = `
          <div class="alert alert-warning">
            <i class="bi bi-exclamation-triangle"></i> No YOLO models found in the models directory. 
            Please add some models to use the application.
          </div>
        `;

        document.getElementById("modelInfo").innerHTML = `
          <div class="alert alert-danger">
            <i class="bi bi-exclamation-triangle-fill"></i> No models available!
            <p>Place your YOLO model files (.pt) in the "models" directory to use the application.</p>
          </div>
        `;
      }
    })
    .catch((error) => {
      document.getElementById("modelInfo").innerHTML = `
        <div class="alert alert-danger">
          <i class="bi bi-exclamation-triangle-fill"></i> Error loading model information: ${error}
        </div>
      `;
    });

  // Model search functionality
  modelSearchInput.addEventListener("input", filterModels);
  
  // Clear search button
  clearSearchBtn.addEventListener("click", function() {
    modelSearchInput.value = "";
    filterModels();
    modelSearchInput.focus();
  });

  // Model selection buttons
  selectAllBtn.addEventListener("click", function () {
    document.querySelectorAll(".model-checkbox").forEach((checkbox) => {
      checkbox.checked = true;
      const card = checkbox.closest(".model-card");
      if (card) updateCardSelection(card, true);
    });
    updateSelectedModelCount();
  });

  deselectAllBtn.addEventListener("click", function () {
    document.querySelectorAll(".model-checkbox").forEach((checkbox) => {
      checkbox.checked = false;
      const card = checkbox.closest(".model-card");
      if (card) updateCardSelection(card, false);
    });
    updateSelectedModelCount();
  });

  // Update threshold values
  document
    .getElementById("confThreshold")
    .addEventListener("input", function () {
      document.getElementById("confThresholdValue").textContent =
        this.value;
    });

  document
    .getElementById("iouThreshold")
    .addEventListener("input", function () {
      document.getElementById("iouThresholdValue").textContent =
        this.value;
    });

  // Form submission
  uploadForm.addEventListener("submit", function (e) {
    e.preventDefault();

    // Check if at least one model is selected
    const selectedModels = document.querySelectorAll(
      ".model-checkbox:checked"
    );
    if (selectedModels.length === 0) {
      alert("Please select at least one model for detection");
      return;
    }

    // Check if files are selected
    if (selectedFiles.files.length === 0) {
      alert("Please select at least one image file");
      return;
    }

    const formData = new FormData(this);
    
    // Replace files in formData
    formData.delete('files[]');
    for (let i = 0; i < selectedFiles.files.length; i++) {
      formData.append('files[]', selectedFiles.files[i]);
    }

    // Show processing info
    const fileCount = selectedFiles.files.length;
    processingStatus.textContent = fileCount > 1 
      ? `Processing ${fileCount} images...` 
      : "Processing image...";
      
    // Hide results and show loading
    resultContainer.style.display = "none";
    batchResultsContainer.style.display = "none";
    loading.style.display = "block";
    
    // Show progress bar for multiple files
    if (fileCount > 1) {
      progressContainer.style.display = "block";
      progressBar.style.width = "0%";
      progressBar.textContent = "0%";
    }

    fetch("/predict", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        // Hide loading
        loading.style.display = "none";
        progressContainer.style.display = "none";

        if (data.error) {
          // Check if it's a model compatibility error and use our helper function
          const customErrorMessage = window.handleModelError ? window.handleModelError(data.error) : null;
          
          if (customErrorMessage) {
            // Create a modal to display the error nicely
            const modalId = "errorModal";
            let modalElement = document.getElementById(modalId);
            
            if (!modalElement) {
              // Create the modal if it doesn't exist
              const modalHtml = `
                <div class="modal fade" id="${modalId}" tabindex="-1" aria-labelledby="${modalId}Label" aria-hidden="true">
                  <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                      <div class="modal-header bg-warning">
                        <h5 class="modal-title" id="${modalId}Label"><i class="bi bi-exclamation-triangle"></i> Model Compatibility Issue</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                      </div>
                      <div class="modal-body" id="${modalId}Body">
                      </div>
                      <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                      </div>
                    </div>
                  </div>
                </div>
              `;
              
              const tempDiv = document.createElement('div');
              tempDiv.innerHTML = modalHtml;
              document.body.appendChild(tempDiv.firstElementChild);
              modalElement = document.getElementById(modalId);
            }
            
            // Update modal body
            document.getElementById(`${modalId}Body`).innerHTML = customErrorMessage;
            
            // Show the modal
            const modal = new bootstrap.Modal(modalElement);
            modal.show();
          } else {
            // Use default alert for other errors
            alert("Error: " + data.error);
          }
          return;
        }

        // Determine if it's a single or batch result
        if (data.results && data.results.length > 1) {
          // Batch processing results
          displayBatchResults(data);
        } else if (data.results && data.results.length === 1) {
          // Single file result - show in the single result view
          displaySingleResult(data.results[0], data.models_used);
        } else {
          alert("No valid results returned from the server");
        }
      })
      .catch((error) => {
        loading.style.display = "none";
        progressContainer.style.display = "none";
        alert("Error: " + error);
      });
  });

  // Display single result
  function displaySingleResult(result, modelsUsed) {
    // Update image
    document.getElementById("resultImage").src =
      "data:image/jpeg;base64," + result.image;

    // Update detection count and models used
    document.getElementById("detectionCount").textContent = result.count;
    document.getElementById("modelsUsed").textContent = modelsUsed.length;

    // Update details table
    const tableBody = document.getElementById("detailsTableBody");
    tableBody.innerHTML = "";

    result.predictions.forEach((pred) => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${pred.class_id}</td>
        <td>${pred.confidence.toFixed(3)}</td>
        <td>${pred.model_name}</td>
        <td>[${pred.box.map((b) => b.toFixed(1)).join(", ")}]</td>
      `;
      tableBody.appendChild(row);
    });

    // Show single result container
    resultContainer.style.display = "block";
  }

  // Display batch results
  function displayBatchResults(data) {
    const resultsAccordion = document.getElementById("resultsAccordion");
    resultsAccordion.innerHTML = "";

    document.getElementById("totalProcessedFiles").textContent =
      data.total_files;
    document.getElementById("batchModelsUsed").textContent =
      data.models_used.length;

    data.results.forEach((result, index) => {
      const accordionItem = document.createElement("div");
      accordionItem.className = "accordion-item";

      const headerId = `heading-${index}`;
      const collapseId = `collapse-${index}`;

      // Create header
      const header = document.createElement("h2");
      header.className = "accordion-header";
      header.id = headerId;

      let buttonContent = `${result.filename}`;
      if (result.error) {
        buttonContent += ` <span class="badge bg-danger">Error</span>`;
      } else {
        buttonContent += ` <span class="badge bg-success">${result.count} detections</span>`;
      }

      header.innerHTML = `
        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" 
          data-bs-target="#${collapseId}" aria-expanded="false" aria-controls="${collapseId}">
          ${buttonContent}
        </button>
      `;

      // Create body
      const body = document.createElement("div");
      body.id = collapseId;
      body.className = "accordion-collapse collapse";
      body.setAttribute("aria-labelledby", headerId);

      let bodyContent = "";
      if (result.error) {
        // Check for model compatibility error
        const customErrorMessage = window.handleModelError ? window.handleModelError(result.error) : null;
        
        if (customErrorMessage) {
          bodyContent = `
            <div class="accordion-body">
              ${customErrorMessage}
            </div>
          `;
        } else {
          bodyContent = `
            <div class="accordion-body">
              <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle"></i> Error: ${result.error}
              </div>
            </div>
          `;
        }
      } else {
        // Create detection details
        let detailsTable = `
          <table class="table table-dark table-striped table-sm">
            <thead>
              <tr>
                <th>Class ID</th>
                <th>Confidence</th>
                <th>Model</th>
                <th>Bounding Box</th>
              </tr>
            </thead>
            <tbody>
        `;

        result.predictions.forEach((pred) => {
          detailsTable += `
            <tr>
              <td>${pred.class_id}</td>
              <td>${pred.confidence.toFixed(3)}</td>
              <td>${pred.model_name}</td>
              <td>[${pred.box.map((b) => b.toFixed(1)).join(", ")}]</td>
            </tr>
          `;
        });

        detailsTable += `
            </tbody>
          </table>
        `;

        bodyContent = `
          <div class="accordion-body">
            <div class="text-center mb-3">
              <img src="data:image/jpeg;base64,${result.image}" class="img-fluid" 
                style="max-height: 400px; border: 1px solid #333; border-radius: 5px;" />
            </div>
            <h6>Detection Details</h6>
            <div class="table-responsive">
              ${detailsTable}
            </div>
          </div>
        `;
      }

      body.innerHTML = bodyContent;

      // Assemble accordion item
      accordionItem.appendChild(header);
      accordionItem.appendChild(body);
      resultsAccordion.appendChild(accordionItem);
    });

    // Show batch results container
    batchResultsContainer.style.display = "block";
  }

  // Add this function to the existing file for preventing double-initialization
  function preventVideoInit() {
    // This is just a marker function to prevent re-initialization
    // of video components which we've moved to index.html
    return true;
  }
});
