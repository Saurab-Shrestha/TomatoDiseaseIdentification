$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').text(' Result:  ' + data);
                console.log('Success!');
            },
        });
    });

});


//onscroll fixed top navigation
$(function () {
    $(window).scroll(function () {
      if ($(this).scrollTop() >= 60) {
        $(".navigation").addClass("sticky-md-top sticky-navigation shadow");
      } else {
        $(".navigation").removeClass("sticky-md-top sticky-navigation shadow");
      }
    });
  });
  
  //dark theme toggle
  $(document).ready(function () {
    $(".darkthemebtn").click(function () {
      $("body").toggleClass("dark-theme");
    });
  });
  
  //slider
  var myCarousel = document.querySelector("#myCarousel");
  var carousel = new bootstrap.Carousel(myCarousel, {
    interval: 2000,
    wrap: false
  });
  
  //back to top button
  $(function () {
    $(window).scroll(function () {
      if ($(this).scrollTop() >= 60) {
        $(".back-to-top").addClass("d-block");
      } else {
        $(".back-to-top").removeClass("d-block");
      }
    });
    $(".back-to-top").click(function () {
      $("html").scrollTop(0);
    });
  });
  

var dropZone = document.getElementById('drop-zone');
var uploadForm = document.getElementById('upload-form');

dropZone.addEventListener('dragover', function(event) {
    event.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', function(event) {
    event.preventDefault();
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', function(event) {
    event.preventDefault();
    dropZone.classList.remove('dragover');

    var files = event.dataTransfer.files;
    uploadForm.querySelector('input[type="file"]').files = files;
});


