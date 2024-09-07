$(document).ready(function() {
    $('#option').change(function() {
        if ($(this).val() === 'B') {
            $('#symbol2-input').show();
        } else {
            $('#symbol2-input').hide();
        }
        
        if (['A', 'B'].includes($(this).val())) {
            $('#date-inputs').show();
        } else {
            $('#date-inputs').hide();
        }
    });

    $('#stock-form').submit(function(e) {
        e.preventDefault();
        
        $.ajax({
            url: '/get_stock_data',
            method: 'POST',
            data: $(this).serialize(),
            success: function(response) {
                if (response.error) {
                    $('#result').html('<p class="error">' + response.error + '</p>');
                } else {
                    Plotly.newPlot('result', JSON.parse(response.plot));
                }
            },
            error: function() {
                $('#result').html('<p class="error">An error occurred. Please try again.</p>');
            }
        });
    });
});