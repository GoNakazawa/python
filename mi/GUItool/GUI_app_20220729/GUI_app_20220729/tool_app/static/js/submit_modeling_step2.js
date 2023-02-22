/*
Submit problemtype to Flask in step2
*/

//Ajax通信でデータ受け渡し
$(function () {

    $('#model_building_button').click(function (event) {
        event.preventDefault();

        //処理中はボタン非活性
        btn_disable();

        //json形式へ変換
        var ptype_val = $('input[name="ptype"]:checked').val();
        var json_data = { "ptype": ptype_val };

        $.ajax({
            data: JSON.stringify(json_data),
            type: 'POST',
            url: '/create_model/' + exp_id,
            dataType: 'json',
            contentType: 'application/json'
        }).done(function (data) {
            console.log(data.OK);
        }).fail(function () {
            location.reload();
        });
        
        return false;
    });
});