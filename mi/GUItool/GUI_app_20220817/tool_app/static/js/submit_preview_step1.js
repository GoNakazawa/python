/*
Submit s3_bucket path, objectives and drop_columns to Flask in step1
*/

//Ajax通信でデータ受け渡し
$(function () {

    $('#btn_submit_preview').click(function (event) {
        event.preventDefault();

        //処理中はボタン非活性
        btn_disable();

        let s3_input_val = $('input[name="s3_input"]').val();
        let s3_input_passwd_val = $('input[name="excel_password"]').val();
        let chem_type_val = $('select[name="chem_type"]').val();
        let s3_master_val = $('input[name="s3_master"]').val();
        let s3_master_passwd_val = $('input[name="master_password"]').val();

        //json形式へ変換
        var json_data = { "s3_input": s3_input_val, "excel_password": s3_input_passwd_val, "chem_type": chem_type_val, "s3_master": s3_master_val, "master_password": s3_master_passwd_val };

        $.ajax({
            data: JSON.stringify(json_data),
            type: 'POST',
            url: '/preview_s3_bucket_data/' + exp_id,
            dataType: 'json',
            contentType: 'application/json'
        }).done(function (data) {
            console.log(data.OK);
        }).always(function () {
            location.reload();
        });
        
        return false;
    });
});